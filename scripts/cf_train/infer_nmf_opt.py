import argparse
import os
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.models.embeddings.nmf_optembed_evol import NmfSearchOpt
from src.trainer import get_cf_trainer
from src.utils import set_seed

set_seed(2023)


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def main(argv: Optional[Sequence[str]] = None):
    config = get_config(argv)
    logger = Logger(**config["logger"])

    # Loading train dataset
    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
    logger.info("Successfully load train dataset")
    train_dataset.describe()
    config["run_test"] = True

    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataset = TestCFGraphDataset(val_dataloader_config["dataset"]["path"])
    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=val_dataloader_config["num_workers"],
    )
    logger.info("Successfully load val dataset")

    checkpoint_folder = os.path.dirname(config["checkpoint_path"])
    os.makedirs(checkpoint_folder, exist_ok=True)

    trainer = get_cf_trainer(
        train_dataset.num_users,
        train_dataset.num_items,
        config,
    )
    model = trainer.model
    logger.info(f"Trainer: {trainer}")

    checkpoint = torch.load(config["checkpoint_path"])
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing_keys:
        logger.debug(f"{missing_keys=}")
    if unexpected_keys:
        logger.debug(f"{unexpected_keys=}")

    # init evol object to validate result
    init_weight_path = config["opt_embed"]["init_weight_path"]
    evol_algo = NmfSearchOpt(
        model,
        n_generations=1,
        population=1,
        n_crossover=1,
        n_mutate=1,
        p_mutate=0.1,
        k=1,
    )
    val_metrics = evol_algo.validate_result(
        init_weight_path,
        val_dataloader,
        train_dataset,
    )

    for key, value in val_metrics.items():
        logger.info(f"{key} - {value:.4f}")

    return


if __name__ == "__main__":
    main()
