"""Training code and utils code for CerpEmbedding.
Create seperate file to avoid circular importing"""

from typing import Dict, Tuple, Union

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.losses import bpr_loss_multi, info_nce
from src.models import IGraphBaseCore
from src.models.lightgcn import LightGCN, SingleLightGCN


def get_prune_and_reg_loss_lightgcn(
    model: Union[LightGCN, SingleLightGCN],
    users: torch.LongTensor,
    pos_items: torch.LongTensor,
    neg_items: torch.LongTensor,
    K: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        model
        users:
        pos_items
        neg_items
    """

    emb: torch.Tensor
    reg_loss: torch.Tensor
    users_unique = torch.unique(users)
    if isinstance(model, LightGCN):
        user_emb = model.user_emb_table(users)
        pos_item_emb = model.item_emb_table(pos_items)
        neg_item_emb = model.item_emb_table(neg_items)

        reg_loss = (
            user_emb.norm(2).pow(2)
            + pos_item_emb.norm(2).pow(2)
            + neg_item_emb.norm(2).pow(2)
        ) / (2 * len(users))

        user_emb = model.user_emb_table(users_unique)
        emb = torch.cat([user_emb, pos_item_emb, neg_item_emb])

    elif isinstance(model, SingleLightGCN):
        indices = torch.cat(
            [users, pos_items + model._num_user, neg_items + model._num_user]
        )
        emb = model.emb_table(indices)
        reg_loss = emb.norm(2).pow(2) / (2 * len(users))

        indices = torch.cat(
            [users_unique, pos_items + model._num_user, neg_items + model._num_user]
        )
        emb = model.emb_table(indices)
    else:
        raise ValueError(f"Not supported model for prune loss {model}")

    prune_loss = -torch.tanh(emb * K).norm(2) ** 2
    return prune_loss, reg_loss


def get_sparsity(model: Union[LightGCN, SingleLightGCN]) -> Tuple[float, int]:
    """Wrapper for get sparsity and num params of model with CerpEmbedding"""
    if isinstance(model, LightGCN):
        _, user_num_params = model.user_emb_table.get_sparsity(True)
        _, item_num_params = model.item_emb_table.get_sparsity(True)
        max_params = model._hidden_size * (model._num_user + model._num_item)
        num_params = user_num_params + item_num_params
        return 1 - num_params / max_params, num_params

    elif isinstance(model, SingleLightGCN):
        sparsity, num_params = model.emb_table.get_sparsity(True)
    else:
        raise ValueError(f"Not supported model for prune loss {model}")
    return sparsity, num_params


def train_epoch_cerp(
    dataloader: DataLoader,
    model: IGraphBaseCore,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    info_nce_weight=0,
    prune_loss_weight=0,
    target_sparsity=0.8,
) -> Dict:
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
        prune_loss=0,
    )
    clip_grad_norm = 100
    for idx, batch in enumerate(dataloader):
        users, pos_items, neg_items = batch
        all_user_emb, all_item_emb = model(adj)

        users = users.to(device)
        pos_items = pos_items.to(device)

        # cast neg_item to shape Batch x Num_Neg
        if isinstance(neg_items, list):
            neg_items = torch.stack(neg_items, dim=1)
        else:
            neg_items = neg_items.unsqueeze(1)
        neg_items = neg_items.to(device)

        user_embs = torch.index_select(all_user_emb, 0, users)
        pos_embs = torch.index_select(all_item_emb, 0, pos_items)

        # shape: batch x num_neg x hidden
        neg_embs = all_item_emb[neg_items]

        rec_loss = bpr_loss_multi(user_embs, pos_embs, neg_embs)

        reg_loss, prune_loss = get_prune_and_reg_loss_lightgcn(
            model, users, pos_items, neg_items.flatten()
        )
        loss_dict["reg_loss"] += reg_loss.item()
        loss_dict["prune_loss"] += prune_loss.item()

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

        loss = (
            rec_loss
            + weight_decay * reg_loss
            + info_nce_loss
            + prune_loss * prune_loss_weight
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        loss_dict["loss"] += loss.item()
        loss_dict["rec_loss"] += rec_loss.item()

        num_sample += users.shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            sparsity, num_params = get_sparsity(model)
            loss_dict["sparsity"] = sparsity
            loss_dict["num_params"] = num_params

            for metric, value in loss_dict.items():
                if metric == "sparsity":
                    msg += f" - {metric}: {value:.2}"
                elif metric == "num_params":
                    msg += f" - {metric}: {value}"
                elif value != 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            logger.info(msg)
            if sparsity >= target_sparsity:
                return loss_dict

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    loss_dict["sparsity"] = sparsity
    loss_dict["num_params"] = num_params

    return loss_dict
