from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
from torch.utils.data import Dataset


class ICTRDataset(Dataset[Tuple[torch.Tensor, float]], ABC):
    field_dims: Iterable[int]

    @abstractmethod
    def pop_info(self):
        ...

    @abstractmethod
    def describe(self):
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """get item function of CTR Dataset

        Returns:
            feature: not added offsets
            label: 1 clicked, 0 not clicked
        """
        ...
