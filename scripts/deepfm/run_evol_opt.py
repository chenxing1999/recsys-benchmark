import argparse
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import get_ctr_dataset
from src.loggers import Logger
from src.models import get_ctr_model

# save logger in evolution logger to logs/evol-logger
from src.models.embeddings.deepfm_opt_embed import evol_search_deepfm
from src.models.embeddings.deepfm_opt_embed import logger as evol_logger
from src.utils import set_seed

evol_logger.add("logs/evol-logger")

set_seed(2023)


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--target-sparsity", default=None, type=float)
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config, args


def main(argv: Optional[Sequence[str]] = None):
    config, args = get_config(argv)
    logger = Logger(**config["logger"])

    # Loading train dataset
    logger.info("Load train dataset...")

    train_dataloader_config = config["train_dataloader"]
    train_dataset = get_ctr_dataset(train_dataloader_config)
    logger.info("Finished load train dataset")

    logger.info("Load val dataset...")
    val_dataloader_config = config["val_dataloader"]
    val_dataloader_config["dataset"]

    # TODO: Refactor later
    val_dataset = get_ctr_dataset(val_dataloader_config, train_dataset.pop_info())
    val_dataset.pop_info()

    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=val_dataloader_config["num_workers"],
    )

    logger.info("Successfully load val dataset")

    model_config = config["model"]
    model = get_ctr_model(train_dataset.field_dims, model_config)
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
    mask, best_ndcg = evol_search_deepfm(
        model,
        num_generations,
        population,
        n_crossover,
        n_mutate,
        p_mutate,
        k,
        val_dataloader,
        train_dataset,
        target_sparsity=args.target_sparsity,
        method=0,
    )

    init_weight_path = config["opt_embed"]["init_weight_path"]
    info = torch.load(init_weight_path)
    info["mask"] = {
        "mask_d": mask,
        "mask_e": model.embedding.get_mask_e(),
    }
    torch.save(info, init_weight_path)


if __name__ == "__main__":
    main()
