from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from ..graph_utils import calculate_sparse_graph_adj_norm
from .base import ICollabRecSys, IGraphBaseCore
from .layers import SparseDropout


class LightGCN(IGraphBaseCore):
    """LightGCN model based on https://arxiv.org/pdf/2002.02126.pdf"""

    def __init__(self, num_user, num_item, num_layers=2, hidden_size=64, p_dropout=0):
        super().__init__()
        self.user_emb_table = nn.Embedding(num_user, hidden_size)
        self.item_emb_table = nn.Embedding(num_item, hidden_size)
        self.num_layers = num_layers
        self._init_normal_weight()
        self._num_user = num_user
        self._num_item = num_item

        self.sparse_dropout = SparseDropout(p_dropout)

    def _init_normal_weight(self):
        nn.init.normal_(self.user_emb_table.weight, std=0.1)
        nn.init.normal_(self.item_emb_table.weight, std=0.1)

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
                self.user_emb_table.weight,
                self.item_emb_table.weight,
            ],
            dim=0,
        )
        matrix = self.sparse_dropout(matrix)

        res = embs
        step = embs
        for _ in range(self.num_layers):
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


class LightGCNInterface(ICollabRecSys):
    def __init__(self, config: LightGCNConfig, graph: Dict[int, List[int]]):
        """
        Args:
            config: LightGCN model config
            graph: Mapping from user to its list of items
        """
        self.core = LightGCN(
            config.num_user,
            config.num_item,
            config.num_layers,
            config.hidden_size,
        )
        self._graph = graph
        self._config = config
        self.refresh_table()

    def refresh_table(self):
        with torch.no_grad():
            preprocessed_adj = calculate_sparse_graph_adj_norm(
                self._graph,
                self._config.num_item,
                self._config.num_user,
            )
            emb_table = self.core.get_emb_table(preprocessed_adj)

        self._emb_table_user = nn.Embedding(
            self.config.num_user, self.config.hidden_size
        )
        self._emb_table_item = nn.Embedding(
            self.config.num_item, self.config.hidden_size
        )

        self._emb_table_user.weight.data = emb_table[: self._config.num_user]
        self._emb_table_item.weight.data = emb_table[self._config.num_user :]

    @torch.no_grad()
    def get_top_k(self, user_id: int, k=5) -> List[int]:
        self.core.eval()

        # Note: This is to keep API format
        # Later this could support elastic size embedding get
        user_id_tensor = torch.tensor([user_id])
        user_emb: torch.tensor = self.get_user_embedding()(user_id_tensor)

        item_embs: nn.Embedding = self.get_item_embedding()
        item_embs = item_embs.weight.data

        # Filter not interacted item
        item_indices = [
            i for i in range(self._config.num_item) if i not in self._graph[user_id]
        ]
        item_indices = torch.tensor(item_indices)

        # Let's start with no batching first. Set batch as all
        item_emb_not_interacted = item_embs[item_indices]
        scores = user_emb @ item_emb_not_interacted.T
        scores = scores.squeeze()

        k = min(len(scores), k)
        results = torch.topk(scores, k=k)
        final_result = item_indices[results[1]]
        return final_result.cpu().tolist()

    def get_item_embedding(self, raw=False):
        if raw:
            return self.core.item_emb_table
        return self._emb_table_item

    def get_user_embedding(self, raw=False):
        if raw:
            return self.core.user_emb_table
        return self._emb_table_user
