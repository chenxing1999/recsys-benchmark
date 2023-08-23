"""Script to run OptEmbed Evolutionary search for LightGCN-based model"""
import argparse
import os
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.models import get_graph_model
from src.models.opt_embed import evol_search_lightgcn


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

    model_config = config["model"]
    model = get_graph_model(
        train_dataset.num_users,
        train_dataset.num_items,
        model_config,
    )

    if config["run_test"]:
        raise NotImplementedError("Will implement later")

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    logger.info(f"Model config: {model_config}")

    checkpoint = torch.load(config["checkpoint_path"])
    model.load_state_dict(checkpoint["state_dict"])

    # TODO: Refactor later
    num_generations = 30
    population = 20
    n_crossover = 10
    n_mutate = 10
    p_mutate = 0.1
    k = 15
    best_item_mask, best_user_mask, best_ndcg = evol_search_lightgcn(
        model,
        num_generations,
        population,
        n_crossover,
        n_mutate,
        p_mutate,
        k,
        val_dataloader,
        train_dataset,
        target_sparsity=0.7,
    )
    nnz = best_item_mask.sum() + best_user_mask.sum()
    total_element = len(best_item_mask) + len(best_user_mask)
    total_element *= model.item_emb_table._hidden_size

    print("Sparsity", 1 - nnz / total_element)


if __name__ == "__main__":
    result = main()
