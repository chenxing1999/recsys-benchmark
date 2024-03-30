from dataclasses import dataclass
from typing import List, Union

import torch

from .base import IGraphBaseCore
from .embeddings import IEmbedding, get_embedding
from .layers import SparseDropout


class LightGCN(IGraphBaseCore):
    """LightGCN model based on https://arxiv.org/pdf/2002.02126.pdf"""

    item_emb_table: IEmbedding
    user_emb_table: IEmbedding

    def __init__(
        self,
        num_user,
        num_item,
        num_layers=2,
        hidden_size=64,
        p_dropout=0,
        embedding_config=None,
    ):
        super().__init__()
        if embedding_config is None:
            embedding_config = {"name": "vanilla"}

        self.embedding_config = embedding_config
        self._init_embedding(num_user, num_item, hidden_size)

        self.num_layers = num_layers

        self._num_user = num_user
        self._num_item = num_item
        self._hidden_size = hidden_size

        if p_dropout > 0:
            self.sparse_dropout = SparseDropout(p_dropout)
        else:
            self.sparse_dropout = torch.nn.Identity(p_dropout)

    def _init_embedding(self, num_user, num_item, hidden_size):
        self.user_emb_table = get_embedding(
            self.embedding_config,
            num_user,
            hidden_size,
            field_name="user",
        )
        self.item_emb_table = get_embedding(
            self.embedding_config,
            num_item,
            hidden_size,
            field_name="item",
        )

    def get_emb_table(self, matrix):
        """
        Args:
            matrix: torch.SparseTensor (num_user + num_item, num_item + num_user)
                denoted as A_tilde in the paper

        Returns
            user_emb: (num_user, hidden_size)
            item_emb: (num_item, hidden_size)
        """

        # embs = E^0 in paper
        # dim: (num_user + num_item) x hidden_size
        embs = torch.cat(
            [
                self.user_emb_table.get_weight(),
                self.item_emb_table.get_weight(),
            ],
            dim=0,
        )
        matrix = self.sparse_dropout(matrix)

        res = embs
        step = embs
        for _ in range(self.num_layers):
            # Where memory peaked
            step = matrix @ step
            res = res + step

        res = res / (self.num_layers + 1)
        return torch.split(res, (self._num_user, self._num_item))

    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        user_emb = self.user_emb_table(users)
        pos_item_emb = self.item_emb_table(pos_items)
        neg_item_emb = self.item_emb_table(neg_items)

        reg_loss = (
            user_emb.norm(2).pow(2)
            + pos_item_emb.norm(2).pow(2)
            + neg_item_emb.norm(2).pow(2)
        ) / (2 * len(users))
        return reg_loss

    def get_embs(self):
        return [
            ("user", self.user_emb_table),
            ("item", self.item_emb_table),
        ]


class SingleLightGCN(IGraphBaseCore):
    """LightGCN model based on https://arxiv.org/pdf/2002.02126.pdf"""

    emb_table: IEmbedding

    def __init__(
        self,
        num_user,
        num_item,
        num_layers=2,
        hidden_size=64,
        p_dropout=0,
        embedding_config=None,
    ):
        super().__init__()
        if embedding_config is None:
            embedding_config = {"name": "vanilla"}

        self.embedding_config = embedding_config
        self._init_embedding(num_user, num_item, hidden_size)

        self.num_layers = num_layers

        self._num_user = num_user
        self._num_item = num_item
        self._hidden_size = hidden_size

        if p_dropout > 0:
            self.sparse_dropout = SparseDropout(p_dropout)
        else:
            self.sparse_dropout = torch.nn.Identity(p_dropout)

    def _init_embedding(self, num_user, num_item, hidden_size):
        self.emb_table = get_embedding(
            self.embedding_config,
            [num_user, num_item],
            hidden_size,
            field_name="user-item",
        )

    def get_emb_table(self, matrix):
        """
        Args:
            matrix: torch.SparseTensor (num_user + num_item, num_item + num_user)
                denoted as A_tilde in the paper

        Returns
            user_emb: (num_user, hidden_size)
            item_emb: (num_item, hidden_size)
        """

        # embs = E^0 in paper
        # dim: (num_user + num_item) x hidden_size
        embs = self.emb_table.get_weight()
        matrix = self.sparse_dropout(matrix)

        res = embs
        step = embs
        for _ in range(self.num_layers):
            # Where memory peaked
            step = matrix @ step
            res = res + step

        res = res / (self.num_layers + 1)
        return torch.split(res, (self._num_user, self._num_item))

    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        indices = torch.cat(
            [users, pos_items + self._num_user, neg_items + self._num_user]
        )
        emb = self.emb_table(indices)
        reg_loss = emb.norm(2).pow(2) / (2 * len(users))

        return reg_loss

    def get_embs(self):
        return [
            ("user-item", self.emb_table),
        ]


@dataclass
class LightGCNConfig:
    num_user: int
    num_item: int

    hidden_size: int = 64
    num_layers: int = 2


def get_sparsity_and_param(model: Union[LightGCN, SingleLightGCN]):
    """Get sparsity of model"""

    max_params: int
    embs: List[IEmbedding]
    if isinstance(model, LightGCN):
        embs = [model.user_emb_table, model.item_emb_table]
        max_params = (model._num_user + model._num_item) * model._hidden_size
    elif isinstance(model, SingleLightGCN):
        embs = [model.emb_table]
        max_params = (model._num_user + model._num_item) * model._hidden_size
    else:
        raise ValueError()

    num_params = 0
    for emb in embs:
        num_params += emb.get_num_params()

    sparsity = 1 - num_params / max_params
    return sparsity, num_params
