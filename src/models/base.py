from abc import ABC
from typing import List

from torch import nn


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
