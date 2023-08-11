import argparse
import gc
import os
from typing import Dict, Optional, Sequence

import loguru
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.criteo import CriteoDataset
from src.loggers import Logger
from src.models.deepfm import DeepFM


def train_epoch(
    dataloader: DataLoader,
    model: DeepFM,
    optimizer,
    device="cuda",
    log_step=10,
    profiler=None,
) -> Dict[str, float]:
    model.train()
    model.to(device)

    loss_dict = dict(loss=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    for idx, batch in enumerate(dataloader):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            loguru.logger.info(msg)
            gc.collect()

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


@torch.no_grad()
def validate_epoch(
    val_loader: DataLoader,
    model: DeepFM,
    device="cuda",
    k=20,
    filter_item_on_train=True,
    profiler=None,
) -> Dict[str, float]:
    """Validate single epoch performance

    Args:
        train_dataset (For getting num_users and norm_adj)
        val_dataloader
        model
        device
        k
        filter_item_on_train: Remove item that user already interacted on train
    Returns:
        "ndcg"
    """

    model.eval()
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion = criterion.to(device)

    log_loss = 0
    all_y_true = []
    all_y_pred = []

    for idx, batch in enumerate(val_loader):
        inputs, labels = batch
        all_y_true.extend(labels.tolist())

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        log_loss += criterion(outputs, labels.float()).item()

        outputs = torch.sigmoid(outputs)
        all_y_pred.extend(outputs.cpu().tolist())

    auc = roc_auc_score(all_y_true, all_y_pred)
    log_loss = log_loss / len(all_y_pred)
    return {
        "auc": auc,
        "log_loss": log_loss,
    }


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def init_profiler(config: Dict):
    """Init PyTorch profiler based on config file

    Args:
        "log_path"
        "schedule"
            "wait"
            "warmup"
            "active"
            "repeat"
        "record_shapes" (default: False)
        "profile_memory" (default: True)
        "with_stack" (default: False)
    """

    log_path = config["log_path"]
    prof_schedule = torch.profiler.schedule(**config["schedule"])
    prof = torch.profiler.profile(
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
        record_shapes=config.get("record_shapes", False),
        profile_memory=config.get("profile_memory", True),
        with_stack=config.get("with_stack", True),
    )
    prof.start()
    return prof


def main(argv: Optional[Sequence[str]] = None):
    config = get_config(argv)
    logger = Logger(**config["logger"])

    # Loading train dataset
    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CriteoDataset(**train_dataset_config)
    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=train_dataloader_config["num_workers"],
    )

    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataset_config = val_dataloader_config["dataset"]

    # TODO: Refactor later
    train_info_to_val = train_dataset.pop_info()
    val_dataset = CriteoDataset(**val_dataset_config, **train_info_to_val)
    val_dataset.pop_info()

    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=val_dataloader_config["num_workers"],
    )

    logger.info("Successfully load val dataset")

    checkpoint_folder = os.path.dirname(config["checkpoint_path"])
    os.makedirs(checkpoint_folder, exist_ok=True)

    model_config = config["model"]
    model = DeepFM(train_dataset.field_dims, **model_config)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"])

        val_metrics = validate_epoch(val_dataloader, model, device)
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    logger.info(f"Model config: {model_config}")

    train_prof, val_prof = None, None
    if config["enable_profile"]:
        train_prof = init_profiler(config["profilers"]["train_profiler"])
        val_prof = init_profiler(config["profilers"]["val_profiler"])

        config["num_epochs"] = 1

    best_auc = 0
    num_epochs = config["num_epochs"]
    try:
        for epoch_idx in range(num_epochs):
            logger.log_metric("Epoch", epoch_idx, epoch_idx)
            train_metrics = train_epoch(
                train_dataloader,
                model,
                optimizer,
                device,
                config["log_step"],
                train_prof,
            )

            train_metrics.update(metrics.get_env_metrics())
            for metric, value in train_metrics.items():
                logger.log_metric(f"train/{metric}", value, epoch_idx)

            if (epoch_idx + 1) % config["validate_step"] == 0:
                val_metrics = validate_epoch(
                    val_dataloader,
                    model,
                    device,
                    filter_item_on_train=True,
                    profiler=val_prof,
                )

                val_metrics.update(metrics.get_env_metrics())

                for key, value in val_metrics.items():
                    logger.log_metric(f"val/{key}", value, epoch_idx)

                if best_auc < val_metrics["auc"]:
                    logger.info("New best, saving model...")
                    best_auc = val_metrics["auc"]

                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "model_config": model_config,
                        "val_metrics": val_metrics,
                    }
                    torch.save(checkpoint, config["checkpoint_path"])
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        pass

        # shared_memory.SharedMemory(train_dataset.name).unlink()
        # shared_memory.SharedMemory(val_dataset.name).unlink()

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
