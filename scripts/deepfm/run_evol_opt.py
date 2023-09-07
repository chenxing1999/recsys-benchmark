import argparse
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.criteo import CriteoDataset, CriteoIterDataset
from src.loggers import Logger
from src.models.deepfm import DeepFM
from src.models.embeddings.deepfm_opt_embed import evol_search_deepfm
from src.utils import set_seed

set_seed(2023)


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def get_dataset_cls(loader_config) -> str:
    num_workers = loader_config.get("num_workers", 0)
    shuffle = loader_config.get("shuffle", False)

    if num_workers == 0 and not shuffle:
        return "iter"
    else:
        return "normal"


NAME_TO_DATASET_CLS = {
    "iter": CriteoIterDataset,
    "normal": CriteoDataset,
}


def main(argv: Optional[Sequence[str]] = None):
    config = get_config(argv)
    logger = Logger(**config["logger"])

    # Loading train dataset
    logger.info("Load train dataset...")

    train_dataloader_config = config["train_dataloader"]
    train_dataset_cls = get_dataset_cls(train_dataloader_config)
    logger.info(f"Train dataset type: {train_dataset_cls}")

    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset_cls = NAME_TO_DATASET_CLS[train_dataset_cls]
    train_dataset = train_dataset_cls(**train_dataset_config)

    logger.info("Load val dataset...")
    val_dataloader_config = config["val_dataloader"]
    val_dataset_config = val_dataloader_config["dataset"]

    # TODO: Refactor later
    val_dataset_cls = get_dataset_cls(val_dataloader_config)
    logger.info(f"Val dataset type: {val_dataset_cls}")
    val_dataset_cls = NAME_TO_DATASET_CLS[val_dataset_cls]
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

    model_config = config["model"]
    model = DeepFM(train_dataset.field_dims, **model_config)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    checkpoint = torch.load(config["checkpoint_path"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # TODO: Refactor later
    num_generations = 30
    population = 20
    n_crossover = 10
    n_mutate = 10
    p_mutate = 0.1
    k = 15
    best_item_mask, best_user_mask, best_ndcg = evol_search_deepfm(
        model,
        num_generations,
        population,
        n_crossover,
        n_mutate,
        p_mutate,
        k,
        val_dataloader,
        train_dataset,
        target_sparsity=None,
        naive=False,
    )


if __name__ == "__main__":
    main()
