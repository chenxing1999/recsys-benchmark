import argparse
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.models import load_graph_model
from src.trainer.lightgcn import validate_epoch
from src.utils import prune, set_seed

set_seed(2023)


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
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)

    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path

    if args.use_test_dataset:
        config["run_test"] = True
    return config, args.prune_ratio


def main(argv: Optional[Sequence[str]] = None):
    config, prune_ratio = get_config(argv)

    # Loading train dataset
    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
    logger.info("Successfully load train dataset")
    train_dataset.describe()

    if config["run_test"]:
        logger.info("Load test dataset...")
        val_dataloader_config = config["test_dataloader"]
    else:
        logger.info("Load val dataset...")
        val_dataloader_config = config["val_dataloader"]

    val_dataset = TestCFGraphDataset(val_dataloader_config["dataset"]["path"])
    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=val_dataloader_config["num_workers"],
    )
    logger.info("Successfully load dataset")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load checkpoint
    checkpoint_path = config["checkpoint_path"]
    model = load_graph_model(checkpoint_path)
    model.to(device)
    if prune_ratio > 0:
        state = prune(model.state_dict(), prune_ratio)
        model.load_state_dict(state)

    val_metrics = validate_epoch(
        train_dataset, val_dataloader, model, device, metrics=["ndcg", "recall"]
    )
    for key, value in val_metrics.items():
        logger.info(f"{key} - {value:.4f}")

    num_params = sum(torch.nonzero(p).size(0) for p in model.parameters())
    logger.info(f"Num Params: {num_params}")


if __name__ == "__main__":
    main()
