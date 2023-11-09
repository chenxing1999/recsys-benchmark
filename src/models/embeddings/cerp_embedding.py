"""Copy and edited from the original CERP code"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from src.losses import bpr_loss_multi, info_nce
from src.models import IGraphBaseCore

from .base import IEmbedding


class CerpEmbedding(IEmbedding):
    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str] = None,
        bucket_size: int = 8000,
        threshold_init: float = -100.0,
        threshold_init_method="all-ones",
    ):
        """
        Args:


        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in [None, "sum", "mean", "max"]

        num_item = sum(field_dims)

        self._field_dims = field_dims
        self._mode = mode

        self.p_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )
        self.q_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )

        nn.init.xavier_uniform_(self.p_weight)
        nn.init.xavier_uniform_(self.q_weight)

        self.q_threshold = self.init_threshold(
            "element-wise",
            threshold_init,
            row_size=bucket_size,
            col_size=hidden_size,
            threshold_init_method=threshold_init_method,
        )
        self.p_threshold = self.init_threshold(
            "element-wise",
            threshold_init,
            row_size=bucket_size,
            col_size=hidden_size,
            threshold_init_method=threshold_init_method,
        )

        self._num_item = num_item
        self._hidden_size = hidden_size

        # Q's avg entities per row = \ceiling(#entities / bucket size)
        self._bucket_size = bucket_size
        self.q_entity_per_row = int(np.ceil(self._num_item / self._bucket_size))

    @staticmethod
    def init_threshold(
        threshold_type: Literal["global", "element-wise"],
        init: float,
        row_size: int,
        col_size: int,
        threshold_init_method: Literal[
            "uniform", "normal", "power_law", "xavier_uniform", "all_ones"
        ] = "all_ones",
    ) -> nn.Parameter:
        requires_scaling = True

        if threshold_type == "global":
            mat = torch.ones(1)
            if threshold_init_method == "uniform":
                mat = mat * torch.rand(1)
                requires_scaling = False
            elif threshold_init_method == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0.0, std=1.0, size=(1,))
            elif threshold_init_method == "power_law":
                a = 0.5
                long_tail_val = torch.tensor(powerlaw.ppf(np.random.random((1,)), a=a))
                mat = mat * long_tail_val
            elif threshold_init_method == "xavier_uniform":
                raise NotImplementedError
            else:
                requires_scaling = False

            if requires_scaling:
                # apply sigmoid to scale this
                mat = torch.sigmoid(mat)

            mat = mat * init
            s = nn.Parameter(mat)
        elif threshold_type == "element-wise":
            mat = torch.ones([row_size, col_size])
            if threshold_init_method == "uniform":
                mat = mat * torch.nn.init.uniform_(torch.zeros((row_size, col_size)))
            elif threshold_init_method == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0.0, std=1.0, size=mat.shape)
            elif threshold_init_method == "power_law":
                a = 0.5
                long_tail_val = torch.tensor(
                    powerlaw.ppf(np.random.random(size=mat.shape), a=a)
                ).float()
                mat = mat * long_tail_val
            elif threshold_init_method == "xavier_uniform":
                mat = mat * nn.init.xavier_uniform_(torch.zeros(size=mat.shape))
            else:
                requires_scaling = False

            if requires_scaling:
                # apply min-max scaling to squeeze
                mat_min, _ = mat.min(dim=1, keepdim=True)
                mat_max, _ = mat.max(dim=1, keepdim=True)
                mat = (mat - mat_min) / (mat_max - mat_min)

            assert (0 <= mat).all() and (1 >= mat).all()
            mat = init * mat
            s = nn.Parameter(mat)
        else:
            raise ValueError("Invalid threshold_type: {}".format(threshold_type))
        return s

    def apply_pruning(self):
        self.sparse_q_weight = torch.sign(self.q_weight) * torch.relu(
            torch.abs(self.q_weight) - torch.sigmoid(self.q_threshold)
        )
        self.sparse_p_weight = torch.sign(self.p_weight) * torch.relu(
            torch.abs(self.p_weight) - torch.sigmoid(self.p_threshold)
        )

    def forward(self, x):
        self.apply_pruning()

        q_idx = torch.div(x, self.q_entity_per_row, rounding_mode="trunc")
        p_idx = x % self._bucket_size

        # get user, items' corresponding embedding vectors in Q, R matrices
        batch_Q_v = F.embedding(q_idx, self.sparse_q_weight)
        batch_P_v = F.embedding(p_idx, self.sparse_p_weight)

        return batch_Q_v + batch_P_v

    def get_sparsity(self, get_n_params=False):
        total_params = self._num_item * self._hidden_size
        self.apply_pruning()

        n_params = 0
        for w in [self.sparse_p_weight, self.sparse_q_weight]:
            n_params += torch.nonzero(w).size(0)

        if get_n_params:
            return (1 - n_params / total_params), n_params
        else:
            return 1 - n_params / total_params

    def get_weight(self):
        self.apply_pruning()

        device = self.sparse_p_weight.device
        all_idxes = torch.arange(self._num_item, device=device)

        return self(all_idxes)


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

        reg_loss = 0
        # if weight_decay > 0:
        reg_loss, prune_loss = model.get_reg_loss(users, pos_items, neg_items)
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

            sparsity, num_params = model.emb_table.get_sparsity(True)
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

    sparsity, num_params = model.emb_table.get_sparsity(True)
    loss_dict["sparsity"] = sparsity
    loss_dict["num_params"] = num_params

    return loss_dict


def prune_loss(emb, K: int = 100):
    return -torch.tanh(emb * K).norm(2) ** 2
