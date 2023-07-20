import argparse
import os
from typing import Dict, List, Optional, Sequence, Set, Union

import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.losses import bpr_loss
from src.models.lightgcn import LightGCN


def train_epoch(
    dataloader: DataLoader,
    model: LightGCN,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
):
    # TODO: Make this to function later lol
    adj = dataloader.dataset._norm_adj
    num_users = dataloader.dataset.num_users

    model.train()
    model.to(device)
    adj = adj.to(device)

    cum_loss = 0
    num_sample = 0
    for idx, batch in enumerate(dataloader):
        users, pos_items, neg_items = batch
        embs = model.get_emb_table(adj)

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        user_embs = torch.index_select(embs, 0, users)
        pos_embs = torch.index_select(embs, 0, pos_items + num_users)
        neg_embs = torch.index_select(embs, 0, neg_items + num_users)

        loss = bpr_loss(user_embs, pos_embs, neg_embs)

        reg_loss = 0
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)

        loss = loss + weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item() * users.shape[0]
        num_sample += users.shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            print(
                "Idx: ",
                idx,
                "- Loss:",
                cum_loss / num_sample,
                "- RegLoss",
                reg_loss.item(),
            )

        if profiler:
            profiler.step()

    return cum_loss / num_sample


@torch.no_grad()
def validate_epoch(
    train_dataset: CFGraphDataset,
    val_loader: DataLoader,
    model: LightGCN,
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
    # TODO: Make this to function later lol
    adj = train_dataset._norm_adj
    num_users = train_dataset.num_users
    graph = train_dataset.get_graph()

    model.eval()
    model = model.to(device)
    adj = adj.to(device)

    # num_user + num_item, hidden_dim
    embs = model.get_emb_table(adj)
    user_embs = embs[:num_users]
    item_embs = embs[num_users:]

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

    logger.info("Load train dataset...")
    train_dataset_config = config["train_dataset"]
    train_dataset = CFGraphDataset(train_dataset_config["path"])
    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataset_config["batch_size"],
        shuffle=True,
        num_workers=train_dataset_config["num_workers"],
    )

    logger.info("Load val dataset...")
    val_dataset_config = config["test_dataset"]
    val_dataset = TestCFGraphDataset(val_dataset_config["path"])
    val_dataloader = DataLoader(
        val_dataset,
        val_dataset_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=4,
    )
    logger.info("Successfully load test dataset")

    checkpoint_folder = os.path.dirname(config["checkpoint_path"])
    os.makedirs(checkpoint_folder, exist_ok=True)

    model_config = config["model"]
    model = LightGCN(
        train_dataset.num_users,
        train_dataset.num_items,
        model_config["num_layers"],
        model_config["hidden_size"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Model config: {model_config}")

    train_prof, val_prof = None, None
    if config["enable_profile"]:
        train_prof = init_profiler(config["profilers"]["train_profiler"])
        val_prof = init_profiler(config["profilers"]["train_profiler"])

        config["num_epochs"] = 1

    best_ndcg = 0
    num_epochs = config["num_epochs"]
    val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
    for epoch_idx in range(num_epochs):
        logger.log_metric("Epoch", epoch_idx, epoch_idx)
        loss = train_epoch(
            train_dataloader,
            model,
            optimizer,
            device,
            config["log_step"],
            config["weight_decay"],
            train_prof,
        )
        logger.log_metric("train/loss", loss, epoch_idx)
        if epoch_idx % config["validate_step"] == 0:
            val_metrics = validate_epoch(
                train_dataset,
                val_dataloader,
                model,
                device,
                filter_item_on_train=True,
                profiler=val_prof,
            )
            for key, value in val_metrics.items():
                logger.log_metric(f"val/{key}", value, epoch_idx)

            if best_ndcg < val_metrics["ndcg"]:
                logger.info("New best, saving model...")
                best_ndcg = val_metrics["ndcg"]

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                }
                torch.save(checkpoint, config["checkpoint_path"])

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
