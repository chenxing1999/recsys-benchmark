import torch
from torch import nn


class SparseDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        """
        Args:
            p: probability of an element to be zeroed. Default: 0.5
            inplace
        """
        self._dropout = nn.Dropout(p, inplace)

    def forward(self, matrix: torch.Tensor):
        values = matrix.values()
        indices = matrix.indices()

        values = self._dropout(values)
        return torch.sparse_coo_tensor(indices, values, matrix.size())
