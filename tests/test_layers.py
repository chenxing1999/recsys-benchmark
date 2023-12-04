import torch

from src.models.layers import SparseDropout


def simple_sparse_dropout_test():
    module = SparseDropout(0.2)
    matrix = torch.eye(5)

    assert module(matrix.to_sparse_csr()).layout == torch.sparse_csr
    assert module(matrix.to_sparse_coo()).layout == torch.sparse_coo
