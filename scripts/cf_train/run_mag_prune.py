import argparse
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.trainer import get_cf_trainer
from src.trainer.nmf import NeuMFTrainer
from src.utils import prune, set_seed

set_seed(2023)


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "-p",
        "--prune-ratio",
        type=float,
        default=0,
        help="Prune percent of model",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        type=str,
        help="Path to checkpoint path",
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="binary or all or none",
        default="none",
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

    return config, args


def get_v(
    trainer: NeuMFTrainer,
    prune_ratio: float,
    num_min_item,
    val_dataloader,
    train_dataset,
    device="cuda",
    get_num_params=False,
):
    model = trainer.model
    state = model.state_dict()
    ori_state = {}
    emb_state = {}

    for k, v in state.items():
        ori_state[k] = v.clone()

        if "emb_table" in k:
            emb_state[k] = v

    new_emb_state = prune(emb_state, prune_ratio, num_min_item)
    state.update(emb_state)
    model.load_state_dict(state)

    val_metrics = trainer.validate_epoch(
        train_dataset,
        val_dataloader,
        metrics=["ndcg", "recall"],
    )
    if get_num_params:
        num_params = 0
        for k, v in new_emb_state.items():
            num_params += torch.count_nonzero(v).item()

        val_metrics["num_params"] = num_params

    for key, value in val_metrics.items():
        logger.info(f"{num_min_item} - {key} - {value:.4f}")

    # restore
    model.load_state_dict(ori_state)

    return val_metrics["ndcg"]


def bin_search(
    trainer,
    prune_ratio,
    hidden_size,
    val_dataloader,
    train_dataset,
    device="cuda",
) -> int:
    max_min_item = int(hidden_size * (1 - prune_ratio))
    v = [None] * (max_min_item + 3)
    v[0] = v[-1] = float("-inf")

    v[1] = get_v(
        trainer,
        prune_ratio,
        0,
        val_dataloader,
        train_dataset,
        device,
    )

    left = 1
    right = max_min_item + 1
    while left <= right:
        num_min_item = (left + right) // 2
        logger.info(f"{num_min_item=} - {left=} - {right=}")

        if v[num_min_item] is None:
            v[num_min_item] = get_v(
                trainer,
                prune_ratio,
                num_min_item - 1,
                val_dataloader,
                train_dataset,
                device,
            )

        if v[1] > v[num_min_item]:
            right = num_min_item - 1
            continue

        if v[num_min_item - 1] is None:
            v[num_min_item - 1] = get_v(
                trainer,
                prune_ratio,
                num_min_item - 2,
                val_dataloader,
                train_dataset,
                device,
            )

        if v[num_min_item + 1] is None:
            v[num_min_item + 1] = get_v(
                trainer,
                prune_ratio,
                num_min_item,
                val_dataloader,
                train_dataset,
                device,
            )

        left_v, mid_v, right_v = v[num_min_item - 1 : num_min_item + 2]

        if left_v < mid_v < right_v:
            left = num_min_item + 1
        elif left_v > mid_v > right_v:
            right = num_min_item - 1
        else:
            print(v)
            return num_min_item

    print(v)
    return 1


def run_all(
    trainer,
    prune_ratio,
    hidden_size,
    val_dataloader,
    train_dataset,
    device="cuda",
) -> int:
    max_min_item = int(hidden_size * (1 - prune_ratio))
    v = []
    for i in range(max_min_item + 1):
        v.append(get_v(trainer, prune_ratio, i, val_dataloader, train_dataset, device))

    return (torch.tensor(v).argmax() + 1).item()


def main(argv: Optional[Sequence[str]] = None):
    config, args = get_config(argv)

    prune_ratio = args.prune_ratio

    # Loading train dataset
    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
    logger.info("Successfully load train dataset")
    train_dataset.describe()

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
    trainer = get_cf_trainer(
        train_dataset.num_users,
        train_dataset.num_items,
        config,
    )
    model = trainer.model
    checkpoint_path = config["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    if args.mode == "binary":
        result = bin_search(
            trainer, prune_ratio, 32, val_dataloader, train_dataset, device
        )
    elif args.mode == "all":
        result = run_all(
            trainer, prune_ratio, 32, val_dataloader, train_dataset, device
        )
    elif args.mode == "none":
        print(get_v(trainer, prune_ratio, 0, val_dataloader, train_dataset, device))
        result = 1

    print(result)

    logger.info("Load test dataset...")
    test_dataloader_config = config["test_dataloader"]
    test_dataset = TestCFGraphDataset(test_dataloader_config["dataset"]["path"])
    test_loader = DataLoader(
        test_dataset,
        test_dataloader_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=test_dataloader_config["num_workers"],
    )

    get_v(
        trainer,
        prune_ratio,
        result - 1,
        test_loader,
        train_dataset,
        device,
        get_num_params=True,
    )


if __name__ == "__main__":
    main()
