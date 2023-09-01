import torch
from typing import Dict
import numpy as np
import random

def prune(state: Dict[str, torch.Tensor], p: float):
    """ Prune state dict based on L2 norm to p sparsity """
    for name, weight in state.items():
        assert len(weight.shape) == 2
        h = weight.shape[1]
        l2 = (weight * weight).flatten()
        indices = torch.argsort(l2)
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
