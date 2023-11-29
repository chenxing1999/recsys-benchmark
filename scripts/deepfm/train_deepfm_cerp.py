import argparse
import os
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.criteo import get_dataset_cls
from src.loggers import Logger
from src.models.deepfm import DeepFM, save_model_checkpoint
from src.trainer.deepfm import train_epoch_cerp, validate_epoch
from src.utils import set_seed

set_seed(2023)
TARGET_SPARSITY = 1300


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
    train_dataset_cls = get_dataset_cls(train_dataloader_config)
    logger.info(f"Train dataset type: {train_dataset_cls}")

    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = train_dataset_cls(**train_dataset_config)

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
    val_dataset_config = val_dataloader_config["dataset"]

    # TODO: Refactor later
    val_dataset_cls = get_dataset_cls(val_dataloader_config)
    logger.info(f"Val dataset type: {val_dataset_cls}")
    train_info_to_val = train_dataset.pop_info()

    val_dataset = val_dataset_cls(**val_dataset_config, **train_info_to_val)
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
    clip_grad = 100

    cerp_config = config.get("cerp", {"gamma_init": 1.0, "gamma_decay": 0.5})
    gamma_init = cerp_config["gamma_init"]
    gamma_decay = cerp_config["gamma_decay"]
    target_sparsity = cerp_config["target_sparsity"]
    trial_checkpoint = cerp_config["trial_checkpoint"]

    save_model_checkpoint(model, trial_checkpoint, "initial")
    try:
        for epoch_idx in range(num_epochs):
            logger.log_metric("Epoch", epoch_idx, epoch_idx)
            train_metrics = train_epoch_cerp(
                train_dataloader,
                model,
                optimizer,
                device,
                config["log_step"],
                train_prof,
                clip_grad,
                target_sparsity=target_sparsity,
                prune_loss_weight=gamma_init * (gamma_decay**epoch_idx),
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

                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "model_config": model_config,
                        "val_metrics": val_metrics,
                    }
                    torch.save(checkpoint, config["checkpoint_path"])

            if train_metrics["sparsity"] >= target_sparsity:
                logger.info("Found target sparsity")
                save_model_checkpoint(model, trial_checkpoint)
                break

    except KeyboardInterrupt:
        pass

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
