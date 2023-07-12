import argparse
import os
from typing import Dict, List, Optional, Sequence

import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.losses import bpr_loss
from src.models.lightgcn import LightGCN

NOT_INTERACTED = {}
FILTER = 1


def train_epoch(
    dataloader: DataLoader,
    model: LightGCN,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
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

        if log_step and idx % log_step == 0:
            print(
                "Idx: ",
                idx,
                "- Loss:",
                cum_loss / num_sample,
                "- RegLoss",
                reg_loss.item(),
            )

    return cum_loss / num_sample


@torch.no_grad()
def validate_epoch_filter(
    train_dataset: CFGraphDataset,
    val_dataset: TestCFGraphDataset,
    model: LightGCN,
    device="cuda",
    verbose=1,
    k=20,
) -> Dict[str, float]:
    """Validate and filter out item that appear in train for user"""

    # TODO: Make this to function later lol
    adj = train_dataset._norm_adj
    num_users = train_dataset.num_users
    train_dataset.num_items
    graph = train_dataset.get_graph()

    model.train()
    model.to(device)
    adj = adj.to(device)

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
    for idx in tqdm.tqdm(range(len(val_dataset))):
        user, pos_item = val_dataset[idx]
        # Filter not interacted item
        interacted = torch.tensor(graph[user], device=user)

        scores = user_embs[user].unsqueeze(0) @ item_embs.T
        scores = scores.squeeze()
        scores[interacted] = float("-inf")

        y_pred = torch.topk(scores, k)[1]

        all_y_pred.append(y_pred.cpu().tolist())
        all_y_true.append(pos_item)

    ndcg = metrics.get_ndcg(all_y_pred, all_y_true)
    if verbose:
        print("NDCG:", ndcg)
    return {
        "ndcg": ndcg,
    }


@torch.no_grad()
def validate_epoch(
    train_dataset: CFGraphDataset,
    val_loader: DataLoader,
    model: LightGCN,
    device="cuda",
    verbose=1,
    k=20,
) -> Dict[str, float]:
    """Validate single epoch performance"""
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

    pos_items: List[List[int]]
    for users, pos_items in val_loader:
        # scores: batch_user x num_items
        scores = user_embs[users] @ item_embs.T

        if FILTER:
            ind0 = []
            ind1 = []
            for idx, user in enumerate(users):
                user = user.item()
                ind0.extend([idx] * len(graph[user]))
                ind1.extend([item for item in graph[user]])

            scores[ind0, ind1] = float("-inf")

        y_pred = torch.topk(scores, k)
        y_pred = y_pred[1]

        all_y_pred.extend(y_pred.cpu().tolist())
        all_y_true.extend(pos_items)

    ndcg = metrics.get_ndcg(all_y_pred, all_y_true)
    if verbose:
        print("NDCG:", ndcg)
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


def main(argv: Optional[Sequence[str]] = None):
    config = get_config(argv)

    print("Load train dataset...")
    train_dataset_config = config["train_dataset"]
    train_dataset = CFGraphDataset(train_dataset_config["path"])
    print("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataset_config["batch_size"],
        shuffle=True,
        num_workers=train_dataset_config["num_workers"],
    )

    print("Load val dataset...")
    val_dataset_config = config["test_dataset"]
    val_dataset = TestCFGraphDataset(val_dataset_config["path"])
    val_dataloader = DataLoader(
        val_dataset,
        val_dataset_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=4,
    )
    print("Successfully load test dataset")

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

    best_ndcg = 0
    num_epochs = config["num_epochs"]
    val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
    for epoch_idx in range(num_epochs):
        print("Epoch - ", epoch_idx)
        loss = train_epoch(
            train_dataloader,
            model,
            optimizer,
            device,
            config["log_step"],
            config["weight_decay"],
        )
        print("Loss - ", loss)
        val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
        # val_metrics = validate_epoch_filter(train_dataset, val_dataset, model, device)

        if best_ndcg < val_metrics["ndcg"]:
            print("New best, saving model...")
            best_ndcg = val_metrics["ndcg"]

            checkpoint = {
                "state_dict": model.state_dict(),
                "model_config": model_config,
            }
            torch.save(checkpoint, config["checkpoint_path"])


if __name__ == "__main__":
    main()
