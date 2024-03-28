"""Define training and evaluating logic for LightGCN"""
from typing import Dict, List, Optional, Set, Union, cast

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import metrics as metric_utils
from src.dataset.cf_graph_dataset import CFGraphDataset
from src.losses import bpr_loss, info_nce
from src.models import IGraphBaseCore
from src.models.lightgcn import LightGCN, SingleLightGCN, get_sparsity_and_param


def train_epoch(
    dataloader: DataLoader,
    model: IGraphBaseCore,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    info_nce_weight=0,
) -> Dict[str, float]:
    """
    Args:


    Returns: Dict contains following keys:
        loss
        reg_loss
        rec_loss
        cl_loss
    """
    train_dataset = cast(CFGraphDataset, dataloader.dataset)
    adj = train_dataset.get_norm_adj()

    model.train()
    model.to(device)
    adj = adj.to(device)

    num_sample = 0

    loss_dict: Dict[str, float] = dict(
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

        reg_loss: torch.Tensor = torch.tensor(0)
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)
            loss_dict["reg_loss"] += reg_loss.item()

        # Enable SGL-Without Augmentation for faster converge
        info_nce_loss: torch.Tensor = torch.tensor(0)
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
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Validate single epoch performance

    Args:
        train_dataset (For getting num_users and norm_adj)
        val_dataloader
        model
        device
        k
        filter_item_on_train: Remove item that user already interacted on train

        metrics: Only support `ndcg` and `recall`
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

    ndcg: float = 0
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

    if metrics is None:
        ndcg = metric_utils.get_ndcg(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
        }
    elif "ndcg" in metrics and "recall" in metrics:
        ndcg, recall = metric_utils.get_ndcg_recall(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
            "recall": recall,
        }
    else:
        ndcg = metric_utils.get_ndcg(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
        }


def train_epoch_optembed(
    dataloader: DataLoader,
    model: Union[LightGCN, SingleLightGCN],
    optimizers: List[torch.optim.Optimizer],
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    info_nce_weight=0,
    alpha=0,
) -> Dict[str, float]:
    """Custom training logic for LightGCN OptEmbed

    Customization have been done:
        - Add `alpha` (weight for l_s)
        - Use multiple optimizers instead of one.
    """
    from src.models.embeddings.lightgcn_opt_embed import OptEmbed

    train_dataset = cast(CFGraphDataset, dataloader.dataset)
    adj = train_dataset.get_norm_adj()
    assert isinstance(model, (LightGCN, SingleLightGCN))

    model.train()
    model.to(device)
    adj = adj.to(device)

    loss_dict: Dict[str, float] = dict(
        loss=0,
        reg_loss=0,
        rec_loss=0,
        cl_loss=0,
        loss_s=0,
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

        reg_loss = torch.tensor(0)
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)
            loss_dict["reg_loss"] += reg_loss.item()

        # Enable SGL-Without Augmentation for faster converge
        info_nce_loss = torch.tensor(0)
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
        if isinstance(model, LightGCN):
            # typehint for mypy
            model.user_emb_table = cast(OptEmbed, model.user_emb_table)
            model.item_emb_table = cast(OptEmbed, model.item_emb_table)

            loss_s = model.user_emb_table.get_l_s()
            loss_s = loss_s + model.item_emb_table.get_l_s()
        elif isinstance(model, SingleLightGCN):
            # typehint for mypy
            model.emb_table = cast(OptEmbed, model.emb_table)

            loss_s = model.emb_table.get_l_s()
        else:
            raise ValueError()
        loss = loss + alpha * loss_s

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        loss_dict["loss_s"] += loss_s.item()
        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"
            sparsity, n_params = get_sparsity_and_param(model)

            msg += f" - {sparsity=:.4f} - {n_params=}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            logger.info(msg)

            # DEBUG CODE ---
            # t_param = model.item_emb_table._mask_e_module._t_param
            # print(
            #     f"Threshold --- "
            #     f"Max: {t_param.max()}"
            #     f"- Min: {t_param.min()}"
            #     f"- Mean: {t_param.mean()}"
            # )
            # norm = model.item_emb_table._weight.norm(1, dim=1)
            # print(
            #     f"Norm --- "
            #     f"Max: {norm.max()}"
            #     f"- Min: {norm.min()}"
            #     f"- Mean: {norm.mean()}"
            # )

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    sparsity, n_params = get_sparsity_and_param(model)
    loss_dict["sparsity"] = sparsity
    loss_dict["n_params"] = n_params
    return loss_dict


def train_epoch_pep(
    dataloader: DataLoader,
    model: IGraphBaseCore,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    info_nce_weight=0,
    target_sparsity=0,
) -> Dict[str, float]:
    """
    Args:


    Returns: Dict contains following keys:
        loss
        reg_loss
        rec_loss
        cl_loss
    """
    from src.models.lightgcn import get_sparsity_and_param

    train_dataset = cast(CFGraphDataset, dataloader.dataset)
    adj = train_dataset.get_norm_adj()

    model.train()
    model.to(device)
    adj = adj.to(device)

    num_sample = 0

    loss_dict: Dict[str, float] = dict(
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

        reg_loss: torch.Tensor = torch.tensor(0)
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)
            loss_dict["reg_loss"] += reg_loss.item()

        # Enable SGL-Without Augmentation for faster converge
        info_nce_loss: torch.Tensor = torch.tensor(0)
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
        loss_dict["rec_loss"] += rec_loss.item()

        num_sample += users.shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            sparsity, num_params = get_sparsity_and_param(model)
            msg += f" - sparsity: {sparsity:.2f} - num_params: {num_params}"

            loss_dict["sparsity"] = sparsity
            loss_dict["num_params"] = num_params
            for metric, value in loss_dict.items():
                if metric in ["sparsity", "num_params"]:
                    continue
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            logger.info(msg)
            if sparsity > target_sparsity:
                logger.info("Found target sparsity")
                break

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        if metric in ["sparsity", "num_params"]:
            continue
        loss_dict[metric] = avg

    return loss_dict
