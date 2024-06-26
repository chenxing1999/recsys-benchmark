from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn


class IGraphBaseCore(nn.Module):
    """LightGCN model based on https://arxiv.org/pdf/2002.02126.pdf"""

    @abstractmethod
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

    @abstractmethod
    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        ...

    def forward(self, matrix):
        return self.get_emb_table(matrix)

    @abstractmethod
    def get_embs(self) -> List[Tuple[str, nn.Module]]:
        """Return pairs of (name, emb_module)"""
        ...


class ICollabRecSys(ABC):
    """Interface for Collaborated-based Recommendation System"""

    @abstractmethod
    def get_top_k(self, user_id, k=5) -> List[int]:
        """Get top k recommendation for user based on id"""
        ...

    @abstractmethod
    def get_item_embedding(self) -> nn.Module:
        """Get item embedding module"""
        ...

    @abstractmethod
    def get_user_embedding(self) -> nn.Module:
        """Get user embedding module"""
        ...


class ISessRecSys(ABC):
    """Interface for Session-based Recommendation System"""

    @abstractmethod
    def get_top_k(self, items: List[int], k=5) -> List[int]:
        """Predict next item based on list item input"""
        ...

    @abstractmethod
    def get_item_embedding(self) -> nn.Module:
        """Get item embedding module"""
        ...
