from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import IEmbedding, get_embedding


class ModelFlag(IntEnum):
    MLP = 1
    GMF = 2
    NMF = 3


class NeuMF(nn.Module):
    """NeuMF implementation


    Note:
    - Assume GMF and MLP have same emb_size
    - Alpha fixed to 0.5 for two model
    """

    def __init__(
        self,
        num_user,
        num_item,
        emb_size: int = 64,
        hidden_sizes: Optional[List[int]] = None,
        p_dropout=0,
        embedding_config=None,
        cache_inference=False,
    ):
        super().__init__()
        if embedding_config is None:
            embedding_config = {"name": "vanilla"}

        assert emb_size % 2 == 0

        # bit 0: mlp, bit 1: gmf
        self.flag = ModelFlag.NMF

        self._num_user = num_user
        self._num_item = num_item
        self._emb_size = emb_size

        self._gmf = _GMF(
            num_user,
            num_item,
            emb_size // 2,
            embedding_config,
            cache_inference,
        )

        self._mlp = _MLP(
            num_user,
            num_item,
            emb_size // 2,
            hidden_sizes,
            p_dropout,
            embedding_config,
            cache_inference,
        )

    def update_weight(self, alpha):
        self._gmf.gmf_fc.weight.data *= 1 - alpha
        self._gmf.gmf_fc.bias.data *= 1 - alpha

        self._mlp.mlp_fc.weight.data *= alpha
        self._mlp.mlp_fc.bias.data *= alpha

    def forward(self, users, items):
        """
        Args:
            users: torch.LongTensor - shape: Batch x K or Batch
            items: torch.LongTensor - shape: Batch x K or Batch

        Returns:
            out: torch.FloatTensor - shape: Batch x K or Batch
        """
        y_mlp = 0
        y_gmf = 0

        if self.mlp_flag():
            y_mlp = self._mlp(users, items)

        if self.gmf_flag():
            y_gmf = self._gmf(users, items)

        return y_mlp + y_gmf

    def mlp_flag(self):
        return self.flag & ModelFlag.MLP

    def gmf_flag(self):
        return self.flag & ModelFlag.GMF

    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        norm = torch.tensor(0)
        if self.mlp_flag():
            mlp_norm = self._mlp.item_emb_table(pos_items).norm(2).pow(2)
            mlp_norm = mlp_norm + self._mlp.item_emb_table(neg_items).norm(2).pow(2)
            mlp_norm = mlp_norm + self._mlp.user_emb_table(users).norm(2).pow(2)
            norm = norm + mlp_norm

        if self.gmf_flag():
            gmf_norm = self._gmf.item_emb_table(pos_items).norm(2).pow(2)
            gmf_norm = gmf_norm + self._gmf.item_emb_table(neg_items).norm(2).pow(2)
            gmf_norm = gmf_norm + self._gmf.user_emb_table(users).norm(2).pow(2)
            norm = norm + gmf_norm

        norm = norm / (2 * len(users))
        return norm

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def get_embs(self) -> List[Tuple[str, IEmbedding]]:
        res = []
        if self.mlp_flag():
            res.extend(
                [
                    ("mlp-user", self._mlp.user_emb_table),
                    ("mlp-item", self._mlp.item_emb_table),
                ]
            )

        if self.gmf_flag():
            res.extend(
                [
                    ("gmf-user", self._gmf.user_emb_table),
                    ("gmf-item", self._gmf.item_emb_table),
                ]
            )

        return res

    def get_prune_loss_tanh(
        self,
        users,
        pos_items,
        neg_items,
        k=100,
    ):
        loss = torch.tensor(0.0, device=users.device)
        if self.mlp_flag():
            emb = self._mlp.item_emb_table(pos_items)
            loss += self._get_prune_loss(emb, k)

            emb = self._mlp.item_emb_table(neg_items)
            loss += self._get_prune_loss(emb, k)

            emb = self._mlp.user_emb_table(users)
            loss += self._get_prune_loss(emb, k)

        if self.gmf_flag():
            emb = self._gmf.item_emb_table(pos_items)
            loss += self._get_prune_loss(emb, k)

            emb = self._gmf.item_emb_table(neg_items)
            loss += self._get_prune_loss(emb, k)

            emb = self._gmf.user_emb_table(users)
            loss += self._get_prune_loss(emb, k)
        return loss

    def _get_prune_loss(self, emb, k):
        """Get pruning loss from CERP"""
        emb = emb * k
        loss = -torch.tanh(emb).norm(2) ** 2
        return loss

    def clear_cache(self):
        self._mlp._user_emb = None
        self._mlp._item_emb = None
        self._gmf._user_emb = None
        self._gmf._item_emb = None


def get_sparsity_and_param(model: NeuMF) -> Tuple[float, int]:
    n_params = 0
    for _, emb in model.get_embs():
        n_params += emb.get_num_params()

    maximum_params = (model.num_user + model.num_item) * model._emb_size

    sparse_rate = 1 - n_params / maximum_params
    return sparse_rate, n_params


class _MLP(nn.Module):
    """MLP Part of NeuMF"""

    def __init__(
        self,
        num_user,
        num_item,
        num_factors: int,
        hidden_sizes: List[int],
        p_dropout: float,
        embedding_config: Dict[str, Any],
        cache_inference=False,
    ):
        super().__init__()
        self.embedding_config = embedding_config
        self._init_embedding(num_user, num_item, num_factors)

        layers = []
        inp_size = num_factors * 2
        for size in hidden_sizes:
            layers.append(nn.Linear(inp_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
            inp_size = size

        self.mlp_fc = nn.Linear(inp_size, 1)
        self.mlp = nn.Sequential(*layers)
        self._init_weight()

        self._cache_inference = cache_inference
        self._user_emb = None
        self._item_emb = None

    def _init_embedding(self, num_user, num_item, hidden_size):
        self.user_emb_table = get_embedding(
            self.embedding_config,
            num_user,
            hidden_size,
            field_name="mlp-user",
        )
        self.item_emb_table = get_embedding(
            self.embedding_config,
            num_item,
            hidden_size,
            field_name="mlp-item",
        )

    def _init_weight(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, users, items):
        """
        Args:
            users: torch.LongTensor - shape: Batch x K or Batch
            items: torch.LongTensor - shape: Batch x K or Batch

        Returns:
            out: torch.FloatTensor - shape: Batch x K or Batch
        """
        if self._cache_inference and not self.training:
            if self._user_emb is None:
                self._user_emb = self.user_emb_table.get_weight()
                self._item_emb = self.item_emb_table.get_weight()

            user_emb = F.embedding(users, self._user_emb)
            item_emb = F.embedding(items, self._item_emb)

        else:
            user_emb = self.user_emb_table(users)
            item_emb = self.item_emb_table(items)

        inputs = torch.cat([user_emb, item_emb], dim=-1)
        out = self.mlp(inputs)
        out = self.mlp_fc(out)
        return out.squeeze(-1)


class _GMF(nn.Module):
    """GMF Part of NeuMF"""

    def __init__(
        self,
        num_user: int,
        num_item: int,
        num_factors: int,
        embedding_config: Dict[str, Any],
        cache_inference: bool = False,
    ):
        """
        Args:
            num_user
            num_item
            num_factors
            embedding_config
            cache_inference: If True, calculate the embedding first
                then infer based on pre-calculated embedding
        """
        super().__init__()
        self.embedding_config = embedding_config
        self._init_embedding(num_user, num_item, num_factors)
        self.gmf_fc = nn.Linear(num_factors, 1)

        self._cache_inference = cache_inference
        self._user_emb = None
        self._item_emb = None

    def _init_embedding(self, num_user, num_item, hidden_size):
        self.user_emb_table = get_embedding(
            self.embedding_config,
            num_user,
            hidden_size,
            field_name="gmf-user",
        )
        self.item_emb_table = get_embedding(
            self.embedding_config,
            num_item,
            hidden_size,
            field_name="gmf-item",
        )

    def forward(self, users, items):
        """
        Args:
            users: torch.LongTensor - shape: Batch x K or Batch
            items: torch.LongTensor - shape: Batch x K or Batch

        Returns:
            out: torch.FloatTensor - shape: Batch x K or Batch
        """
        if self._cache_inference and not self.training:
            if self._user_emb is None:
                self._user_emb = self.user_emb_table.get_weight()
                self._item_emb = self.item_emb_table.get_weight()

            user_emb = F.embedding(users, self._user_emb)
            item_emb = F.embedding(items, self._item_emb)

        else:
            user_emb = self.user_emb_table(users)
            item_emb = self.item_emb_table(items)

        out = user_emb * item_emb
        out = self.gmf_fc(out)

        return out.squeeze(-1)
