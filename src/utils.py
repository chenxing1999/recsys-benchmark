import random
from typing import Dict

import numpy as np
import torch


def prune(
    state: Dict[str, torch.Tensor],
    p: float,
    min_item: int = 0,
):
    """Prune state dict based on L2 norm to p sparsity"""
    for name, weight in state.items():
        assert len(weight.shape) == 2
        n_item, h = weight.shape

        norm = weight.abs()

        topk = norm.topk(min_item, 1)
        norm[
            torch.arange(n_item).unsqueeze(-1).repeat(1, min_item), topk.indices
        ] = float("inf")

        indices = torch.argsort(norm.flatten())
        ori_i = indices // h
        ori_j = indices % h

        num_prune = int(indices.shape[0] * p)
        weight[ori_i[:num_prune], ori_j[:num_prune]] = 0

        state[name] = weight

    return state


def random_prune(
    state: Dict[str, torch.Tensor],
    p: float,
):
    """Pruning model is a random way"""
    for name, weight in state.items():
        assert len(weight.shape) == 2
        n_item, h = weight.shape

        indices = torch.randperm(n_item * h)
        ori_i = indices // h
        ori_j = indices % h

        num_prune = int(indices.shape[0] * p)
        weight[ori_i[:num_prune], ori_j[:num_prune]] = 0

        state[name] = weight

    return state


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_size_tensor(tensor: torch.Tensor) -> int:
    """Return memory size of tensor in bytes"""
    # Note: Tensor.is_sparse return False for tensor.is_sparse_csr
    if tensor.layout == torch.strided:
        return tensor.element_size() * tensor.nelement()
    elif tensor.layout == torch.sparse_csr:
        size = get_size_tensor(tensor.values())
        size += get_size_tensor(tensor.crow_indices())
        size += get_size_tensor(tensor.col_indices())
        return size
    elif tensor.layout == torch.sparse_coo:
        size = get_size_tensor(tensor.values())
        size += get_size_tensor(tensor.indices())
        return size
    elif tensor.layout == torch.sparse_bsr:
        size = get_size_tensor(tensor.values())
        size += get_size_tensor(tensor.crow_indices())
        size += get_size_tensor(tensor.col_indices())
        return size
    else:
        raise NotImplementedError()
