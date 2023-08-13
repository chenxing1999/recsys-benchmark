from dataclasses import dataclass

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

        if p_dropout > 0:
            self.sparse_dropout = SparseDropout(p_dropout)
        else:
            self.sparse_dropout = torch.nn.Identity(p_dropout)

    def _init_embedding(self, num_user, num_item, hidden_size):
        self.user_emb_table = get_embedding(
            self.embedding_config,
            num_user,
            hidden_size,
        )
        self.item_emb_table = get_embedding(
            self.embedding_config,
            num_item,
            hidden_size,
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


@dataclass
class LightGCNConfig:
    num_user: int
    num_item: int

    hidden_size: int = 64
    num_layers: int = 2
