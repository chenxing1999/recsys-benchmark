import argparse
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset import get_ctr_dataset
from src.models import load_ctr_model
from src.models.embeddings.ptq_emb import PTQEmb_Int
from src.trainer.deepfm import validate_epoch


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "--n_bits",
        type=int,
        choices=[16, 32, 8],
        help="N-bits for embedding layer",
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
    parser.add_argument(
        "-n",
        "--num_min_item",
        help="Number of min item per user/item",
        default=0,
        type=int,
    )

    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)

    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path

    if args.use_test_dataset:
        config["run_test"] = True
    return config, args


def main(argv: Optional[Sequence[str]] = None):
    config, args = get_config(argv)

    args.output_path

    # Loading train dataset
    logger.info("Load train dataset...")

    train_dataloader_config = config["train_dataloader"]
    train_dataset = get_ctr_dataset(train_dataloader_config)

    logger.info("Successfully load train dataset")
    train_dataset.describe()

    if config["run_test"]:
        logger.info("Load test dataset...")
        val_dataloader_config = config["test_dataloader"]
    else:
        logger.info("Load val dataset...")
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataloader_config["dataset"]

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
    checkpoint_path = config["checkpoint_path"]
    model = load_ctr_model(config["model"], checkpoint_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load checkpoint
    checkpoint_path = config["checkpoint_path"]
    model.embedding = PTQEmb_Int(None, None, None, checkpoint_path, args.n_bits)
    model.to(device)

    # uncomment this for TTRec
    # model.embedding._tt_emb.warmup = False

    val_metrics = validate_epoch(val_dataloader, model, device)
    for key, value in val_metrics.items():
        logger.info(f"{key} - {value:.4f}")

    num_params = sum(torch.nonzero(p).size(0) for p in model.parameters())
    logger.info(f"Num Params: {num_params}")

    num_params = sum(torch.nonzero(p).size(0) for p in model.embedding.parameters())
    logger.info(f"Num Emb Params: {num_params}")


if __name__ == "__main__":
    main()
