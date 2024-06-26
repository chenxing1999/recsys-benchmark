import argparse
import os
from typing import Dict, Optional, Sequence, Union

import loguru
import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset import get_ctr_dataset
from src.loggers import Logger
from src.models import get_ctr_model
from src.models.dcn import DCN_Mix
from src.models.deepfm import DeepFM
from src.trainer.deepfm import validate_epoch
from src.utils import set_seed

set_seed(2023)
TARGET_SPARSITY = 1300


def train_epoch(
    dataloader: DataLoader,
    model: Union[DeepFM, DCN_Mix],
    optimizer,
    device="cuda",
    log_step=10,
    profiler=None,
    clip_grad=0,
) -> Dict[str, float]:
    """
    Note: log_step is used to check if model's sparsity found or not
    """
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
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            with torch.no_grad():
                sparsity, n_params = model.embedding.get_sparsity(True)
            msg += f" - params: {n_params} - sparsity: {sparsity:.2}"
            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            loguru.logger.info(msg)
            model.embedding.train_callback()

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


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
    train_dataset = get_ctr_dataset(train_dataloader_config)

    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataloader_config["batch_size"],
        shuffle=train_dataloader_config.get("shuffle", False),
        num_workers=train_dataloader_config["num_workers"],
    )

    # Loading val dataset
    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataloader_config["dataset"]

    # TODO: Refactor later
    train_info_to_val = train_dataset.pop_info()

    val_dataset = get_ctr_dataset(val_dataloader_config, train_info_to_val)
    val_dataset.pop_info()
    val_dataset.describe()

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
    model = get_ctr_model(train_dataset.field_dims, model_config)

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

    is_pep = model_config["embedding_config"]["name"] == "pep"
    best_auc = 0
    num_epochs = config["num_epochs"]
    clip_grad = 100
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
                clip_grad,
            )

            train_metrics.update(metrics.get_env_metrics())
            for metric, value in train_metrics.items():
                logger.log_metric(f"train/{metric}", value, epoch_idx)

            if (epoch_idx + 1) % config["validate_step"] == 0:
                val_metrics = validate_epoch(
                    val_dataloader,
                    model,
                    device,
                )

                val_metrics.update(metrics.get_env_metrics())

                for key, value in val_metrics.items():
                    logger.log_metric(f"val/{key}", value, epoch_idx)

                if best_auc < val_metrics["auc"]:
                    logger.info("New best, saving model...")
                    best_auc = val_metrics["auc"]

                    sparsity, n_params = model.embedding.get_sparsity(True)
                    val_metrics["sparsity"] = sparsity

                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "model_config": model_config,
                        "val_metrics": val_metrics,
                    }
                    torch.save(checkpoint, config["checkpoint_path"])

                if is_pep:
                    sparsity, n_params = model.embedding.get_sparsity(True)
                    logger.log_metric("sparsity", sparsity, epoch_idx)
                    logger.info(f"{n_params=}")
                    path = os.path.join(
                        model.embedding.checkpoint_weight_dir, f"{TARGET_SPARSITY}.pth"
                    )
                    emb = model.embedding
                    if len(emb.sparsity) <= emb._cur_min_spar_idx:
                        logger.debug("Found all sparsity")
                        break
                    if n_params <= TARGET_SPARSITY and not os.path.exists(path):
                        logger.info(f"save {TARGET_SPARSITY}")
                        torch.save(model.embedding.state_dict(), path)

    except KeyboardInterrupt:
        pass

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
