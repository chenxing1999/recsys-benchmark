import torch
from torch import nn


class SparseDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        """
        Args:
            p: probability of an element to be zeroed. Default: 0.5
            inplace
        """
        super().__init__()
        self._dropout = nn.Dropout(p, inplace)

    def forward(self, matrix: torch.Tensor):
        if matrix.is_sparse_csr:
            # Matrix CSR
            values = matrix.values()
            values = self._dropout(values)
            crow_indices = matrix.crow_indices
            col_indices = matrix.col_indices
            return torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                values,
                matrix.size(),
            )
        else:
            # Matrix COO
            matrix = matrix.coalesce()
            values = matrix.values()
            indices = matrix.indices()

            values = self._dropout(values)
            return torch.sparse_coo_tensor(indices, values, matrix.size())
