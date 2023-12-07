import argparse
import os
from typing import Dict, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.models import get_graph_model
from src.trainer.lightgcn import train_epoch, validate_epoch
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
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataloader_config["batch_size"],
        shuffle=True,
        num_workers=train_dataloader_config["num_workers"],
    )

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

    # opt_embed specific code
    is_retrain = "retrain" in model_config["embedding_config"]["name"]
    is_opt_embed = "opt_embed" in config
    if is_opt_embed and not is_retrain:
        init_weight_path = config["opt_embed"]["init_weight_path"]
        torch.save(
            {
                "full": model.state_dict(),
            },
            config["opt_embed"]["init_weight_path"],
        )
    elif is_opt_embed:
        init_weight_path = config["opt_embed"]["init_weight_path"]
        info = torch.load(init_weight_path)
        mask = info["mask"]
        keys = model.load_state_dict(info["full"], False)
        length_miss = len(keys[0])
        expected_miss = sum(
            1
            for key in keys[0]
            if key in ["user_emb_table._mask", "item_emb_table._mask"]
        )
        length_miss = length_miss - expected_miss
        assert length_miss == 0, f"There are some keys missing: {keys[0]}"
        model.item_emb_table.init_mask(mask_d=mask["item"]["mask_d"], mask_e=None)
        model.user_emb_table.init_mask(mask_d=mask["user"]["mask_d"], mask_e=None)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        val_metrics = validate_epoch(
            train_dataset, val_dataloader, model, device, metrics=["recall", "ndcg"]
        )
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
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

    best_ndcg = 0
    num_epochs = config["num_epochs"]

    early_stop_count = 0
    early_stop_config = config.get("early_stop_patience", 0)
    warmup = config.get("warmup", 0)
    import time
    start = time.time()
    for epoch_idx in range(1):
        logger.log_metric("Epoch", epoch_idx, epoch_idx)
        train_metrics = train_epoch(
            train_dataloader,
            model,
            optimizer,
            device,
            config["log_step"],
            config["weight_decay"],
            train_prof,
            info_nce_weight=config["info_nce_weight"],
        )

        train_metrics.update(metrics.get_env_metrics())
        for metric, value in train_metrics.items():
            logger.log_metric(f"train/{metric}", value, epoch_idx)

        if (epoch_idx + 1) % config["validate_step"] == 0:
            val_metrics = validate_epoch(
                train_dataset,
                val_dataloader,
                model,
                device,
                filter_item_on_train=True,
                profiler=val_prof,
            )

            val_metrics.update(metrics.get_env_metrics())

            for key, value in val_metrics.items():
                logger.log_metric(f"val/{key}", value, epoch_idx)

            if best_ndcg < val_metrics["ndcg"]:
                logger.info("New best, saving model...")
                best_ndcg = val_metrics["ndcg"]

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "val_metrics": val_metrics,
                    "num_users": train_dataset.num_users,
                    "num_items": train_dataset.num_items,
                }
                torch.save(checkpoint, config["checkpoint_path"])
                early_stop_count = 0
            elif warmup <= epoch_idx:
                early_stop_count += 1
                logger.debug(f"{early_stop_count=}")

                if early_stop_config and early_stop_count > early_stop_config:
                    return

    print(time.time() - start)
    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
