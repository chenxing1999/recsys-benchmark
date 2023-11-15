import argparse
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset.criteo import get_dataset_cls
from src.models.deepfm import DeepFM
from src.trainer.deepfm import validate_epoch
from src.utils import prune


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "-p",
        "--prune-ratio",
        type=float,
        default=0,
        help="Prune percent of model. If is 0, this script become evaluate",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        type=str,
        help="Path to checkpoint path",
        default=None,
    )
    parser.add_argument(
        "--use-test-dataset",
        action="store_true",
        help="Force using test dataset",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to saving pruned checkpoint",
    )

    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)

    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path

    if args.use_test_dataset:
        config["run_test"] = True
    return config, args.prune_ratio, args.output_path


def main(argv: Optional[Sequence[str]] = None):
    config, prune_ratio, output_path = get_config(argv)

    # Loading train dataset
    logger.info("Load train dataset...")

    train_dataloader_config = config["train_dataloader"]
    train_dataset_cls = get_dataset_cls(train_dataloader_config)
    logger.info(f"Train dataset type: {train_dataset_cls}")

    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = train_dataset_cls(**train_dataset_config)

    logger.info("Successfully load train dataset")
    train_dataset.describe()

    if config["run_test"]:
        logger.info("Load test dataset...")
        val_dataloader_config = config["test_dataloader"]
    else:
        logger.info("Load val dataset...")
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataset_config = val_dataloader_config["dataset"]

    # TODO: Refactor later
    val_dataset_cls = get_dataset_cls(val_dataloader_config)
    logger.info(f"Val dataset type: {val_dataset_cls}")
    train_info_to_val = train_dataset.pop_info()

    val_dataset = val_dataset_cls(**val_dataset_config, **train_info_to_val)
    val_dataset.pop_info()

    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=val_dataloader_config["num_workers"],
    )

    logger.info("Successfully load val dataset")
    checkpoint_path = config["checkpoint_path"]
    model = DeepFM.load(checkpoint_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load checkpoint
    checkpoint_path = config["checkpoint_path"]
    model = DeepFM.load(checkpoint_path)
    model.to(device)
    if prune_ratio > 0:
        state = prune(model.embedding.state_dict(), prune_ratio)
        model.embedding.load_state_dict(state)

    val_metrics = validate_epoch(val_dataloader, model, device)
    for key, value in val_metrics.items():
        logger.info(f"{key} - {value:.4f}")

    num_params = sum(torch.nonzero(p).size(0) for p in model.parameters())
    logger.info(f"Num Params: {num_params}")

    if output_path is not None and prune_ratio > 0:
        checkpoint = torch.load(checkpoint_path)
        checkpoint["state_dict"] = state
        torch.save(checkpoint, output_path)


if __name__ == "__main__":
    main()
