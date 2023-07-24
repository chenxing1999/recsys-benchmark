from abc import ABC
from typing import List, Tuple

import torch
from torch import nn


class IGraphBaseCore(nn.Module):
    """LightGCN model based on https://arxiv.org/pdf/2002.02126.pdf"""

    def get_emb_table(self, matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            matrix: torch.SparseTensor (num_user + num_item, num_item + num_user)
                denoted as A_tilde in the paper

        Returns
            user_emb:
            item_emb:

        """
        ...

    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        ...

    def forward(self, matrix):
        return self.get_emb_table(matrix)


class ICollabRecSys(ABC):
    """Interface for Collaborated-based Recommendation System"""

    def get_top_k(self, user_id, k=5) -> List[int]:
        """Get top k recommendation for user based on id"""
        ...

    def get_item_embedding(self) -> nn.Module:
        """Get item embedding module"""
        ...

    def get_user_embedding(self) -> nn.Module:
        """Get user embedding module"""
        ...


class ISessRecSys(ABC):
    """Interface for Session-based Recommendation System"""

    def get_top_k(self, items: List[int], k=5) -> List[int]:
        """Predict next item based on list item input"""
        ...

    def get_item_embedding(self) -> nn.Module:
        """Get item embedding module"""
        ...
