import argparse
import os
from typing import Dict, List, Optional, Sequence, Set, Union

import loguru
import psutil
import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.losses import bpr_loss, info_nce
from src.models import IGraphBaseCore, get_graph_model


def get_env_metrics() -> Dict[str, float]:
    """Utils function for quick benchmark performance

    Returns: a metric dictionary that contains
        various computation wise metrics such as cpu mem and peak CUDA mem

    Note: Copied and modified from RecBole
    """

    memory_used = psutil.Process(os.getpid()).memory_info().rss
    cpu_usage = psutil.cpu_percent(interval=1)
    peak_cuda_mem = torch.cuda.max_memory_allocated()
    cur_cuda_mem = torch.cuda.memory_allocated()

    return {
        "cur_cpu_memory": memory_used,
        "cur_cpu_usage": cpu_usage,
        "cur_cuda_mem": cur_cuda_mem,
        "peak_cuda_mem": peak_cuda_mem,
    }


def train_epoch(
    dataloader: DataLoader,
    model: IGraphBaseCore,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    info_nce_weight=0,
):
    adj = dataloader.dataset.get_norm_adj()

    model.train()
    model.to(device)
    adj = adj.to(device)

    num_sample = 0

    loss_dict = dict(
        loss=0,
        reg_loss=0,
        rec_loss=0,
        cl_loss=0,
    )
    for idx, batch in enumerate(dataloader):
        users, pos_items, neg_items = batch
        all_user_emb, all_item_emb = model.get_emb_table(adj)

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        user_embs = torch.index_select(all_user_emb, 0, users)
        pos_embs = torch.index_select(all_item_emb, 0, pos_items)
        neg_embs = torch.index_select(all_item_emb, 0, neg_items)

        rec_loss = bpr_loss(user_embs, pos_embs, neg_embs)

        reg_loss = 0
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)

        # Enable SGL-Without Augmentation for faster converge
        info_nce_loss = 0
        if info_nce_weight > 0:
            temperature = 0.2

            tmp_user_idx = torch.unique(users)
            tmp_user_embs = torch.index_select(all_user_emb, 0, tmp_user_idx)

            tmp_pos_idx = torch.unique(pos_items)
            tmp_pos_embs = torch.index_select(all_item_emb, 0, tmp_pos_idx)
            view1 = torch.cat([tmp_user_embs, tmp_pos_embs], 0)

            info_nce_loss = info_nce(view1, view1, temperature)

            info_nce_loss = info_nce_loss * info_nce_weight
            loss_dict["cl_loss"] += info_nce_loss.item()

        loss = rec_loss + weight_decay * reg_loss + info_nce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict["loss"] += loss.item()
        loss_dict["reg_loss"] += reg_loss.item()
        loss_dict["rec_loss"] += rec_loss.item()

        num_sample += users.shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            loguru.logger.info(msg)

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


@torch.no_grad()
def validate_epoch(
    train_dataset: CFGraphDataset,
    val_loader: DataLoader,
    model: IGraphBaseCore,
    device="cuda",
    k=20,
    filter_item_on_train=True,
    profiler=None,
) -> Dict[str, float]:
    """Validate single epoch performance

    Args:
        train_dataset (For getting num_users and norm_adj)
        val_dataloader
        model
        device
        k
        filter_item_on_train: Remove item that user already interacted on train
    Returns:
        "ndcg"
    """
    adj = train_dataset.get_norm_adj()
    graph = train_dataset.get_graph()

    model.eval()
    model = model.to(device)
    adj = adj.to(device)

    # num_user + num_item, hidden_dim
    user_embs, item_embs = model.get_emb_table(adj)

    ndcg = 0
    all_y_pred = []
    all_y_true = []

    pos_items: List[Union[Set[int], List[int]]]
    for users, pos_items in val_loader:
        # scores: batch_user x num_items
        scores = user_embs[users] @ item_embs.T

        if filter_item_on_train:
            ind0 = []
            ind1 = []
            for idx, user in enumerate(users.tolist()):
                ind0.extend([idx] * len(graph[user]))
                ind1.extend(graph[user])

            scores[ind0, ind1] = float("-inf")

        y_pred = torch.topk(scores, k)
        y_pred = y_pred[1]

        all_y_pred.extend(y_pred.cpu().tolist())

        all_y_true.extend(pos_items)
        if profiler:
            profiler.step()

    ndcg = metrics.get_ndcg(all_y_pred, all_y_true)
    return {
        "ndcg": ndcg,
    }


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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"])

        val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
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
    val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
    for epoch_idx in range(num_epochs):
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

        train_metrics.update(get_env_metrics())
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

            val_metrics.update(get_env_metrics())

            for key, value in val_metrics.items():
                logger.log_metric(f"val/{key}", value, epoch_idx)

            if best_ndcg < val_metrics["ndcg"]:
                logger.info("New best, saving model...")
                best_ndcg = val_metrics["ndcg"]

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, config["checkpoint_path"])

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
