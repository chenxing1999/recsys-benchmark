import argparse
import os
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset import get_ctr_dataset
from src.loggers import Logger
from src.models.deepfm import DeepFM, get_optimizers
from src.trainer.deepfm import train_epoch, validate_epoch
from src.utils import set_seed

set_seed(2023)


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

    # TODO: Refactor later
    train_info_to_val = train_dataset.pop_info()

    val_dataset = get_ctr_dataset(val_dataloader_config, train_info_to_val)
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

    if "deepfm_optembed_retrain" in config["model"]["embedding_config"]["name"]:
        logger.info("DeepFM OptEmbed Retrain detected")
        init_weight_path = config["opt_embed"]["init_weight_path"]
        info = torch.load(init_weight_path)
        mask = info["mask"]
        keys = model.load_state_dict(info["full"], False)
        assert len(keys[0]) == 0, f"There are some keys missing: {keys[0]}"
        model.embedding.init_mask(mask_d=mask["mask_d"], mask_e=mask["mask_e"])
        logger.info(f"Num params: {model.embedding.get_num_params()}")

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"], False)

        val_metrics = validate_epoch(val_dataloader, model, device)
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    optimizers = get_optimizers(model, config)

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    num_emb_params = 0
    for p in model.embedding.parameters():
        num_emb_params += p.numel()
    logger.log_metric("num_emb_params", num_emb_params)

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
                optimizers,
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
                        "field_dims": train_dataset.field_dims,
                    }
                    torch.save(checkpoint, config["checkpoint_path"])
    except KeyboardInterrupt:
        pass

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
