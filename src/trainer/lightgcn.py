"""Define training and evaluating logic for LightGCN"""
from typing import Dict, List, Set, Union

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset
from src.losses import bpr_loss, info_nce
from src.models import IGraphBaseCore


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
        all_user_emb, all_item_emb = model(adj)

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

            logger.info(msg)

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
    user_embs, item_embs = model(adj)

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
