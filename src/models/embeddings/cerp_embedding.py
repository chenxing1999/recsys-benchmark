"""Copy and edited from the original CERP code"""
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import IEmbedding


class CerpEmbedding(IEmbedding):
    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str] = None,
        bucket_size: int = 8000,
        threshold_init: float = -100.0,
        threshold_init_method="all-ones",
    ):
        """
        Args:


        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in [None, "sum", "mean", "max"]

        num_item = sum(field_dims)

        self._field_dims = field_dims
        self._mode = mode

        self.p_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )
        self.q_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )

        nn.init.xavier_uniform_(self.p_weight)
        nn.init.xavier_uniform_(self.q_weight)

        self.q_threshold = self.init_threshold(
            "element-wise",
            threshold_init,
            row_size=bucket_size,
            col_size=hidden_size,
            threshold_init_method=threshold_init_method,
        )
        self.p_threshold = self.init_threshold(
            "element-wise",
            threshold_init,
            row_size=bucket_size,
            col_size=hidden_size,
            threshold_init_method=threshold_init_method,
        )

        self._num_item = num_item
        self._hidden_size = hidden_size

        # Q's avg entities per row = \ceiling(#entities / bucket size)
        self._bucket_size = bucket_size
        self.q_entity_per_row = int(np.ceil(self._num_item / self._bucket_size))

    @staticmethod
    def init_threshold(
        threshold_type: Literal["global", "element-wise"],
        init: float,
        row_size: int,
        col_size: int,
        threshold_init_method: Literal[
            "uniform", "normal", "xavier_uniform", "all_ones"
        ] = "all_ones",
    ) -> nn.Parameter:
        requires_scaling = True

        if threshold_type == "global":
            mat = torch.ones(1)
            if threshold_init_method == "uniform":
                mat = mat * torch.rand(1)
                requires_scaling = False
            elif threshold_init_method == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0.0, std=1.0, size=(1,))
            elif threshold_init_method == "xavier_uniform":
                raise NotImplementedError
            else:
                requires_scaling = False

            if requires_scaling:
                # apply sigmoid to scale this
                mat = torch.sigmoid(mat)

            mat = mat * init
            s = nn.Parameter(mat)
        elif threshold_type == "element-wise":
            mat = torch.ones([row_size, col_size])
            if threshold_init_method == "uniform":
                mat = mat * torch.nn.init.uniform_(torch.zeros((row_size, col_size)))
            elif threshold_init_method == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0.0, std=1.0, size=mat.shape)
            elif threshold_init_method == "xavier_uniform":
                mat = mat * nn.init.xavier_uniform_(torch.zeros(size=mat.shape))
            else:
                requires_scaling = False

            if requires_scaling:
                # apply min-max scaling to squeeze
                mat_min, _ = mat.min(dim=1, keepdim=True)
                mat_max, _ = mat.max(dim=1, keepdim=True)
                mat = (mat - mat_min) / (mat_max - mat_min)

            assert (0 <= mat).all() and (1 >= mat).all()
            mat = init * mat
            s = nn.Parameter(mat)
        else:
            raise ValueError("Invalid threshold_type: {}".format(threshold_type))
        return s

    def apply_pruning(self):
        self.sparse_q_weight = torch.sign(self.q_weight) * torch.relu(
            torch.abs(self.q_weight) - torch.sigmoid(self.q_threshold)
        )
        self.sparse_p_weight = torch.sign(self.p_weight) * torch.relu(
            torch.abs(self.p_weight) - torch.sigmoid(self.p_threshold)
        )

    def forward(self, x):
        self.apply_pruning()

        q_idx = torch.div(x, self.q_entity_per_row, rounding_mode="trunc")
        p_idx = x % self._bucket_size

        # get user, items' corresponding embedding vectors in Q, R matrices
        batch_Q_v = F.embedding(q_idx, self.sparse_q_weight)
        batch_P_v = F.embedding(p_idx, self.sparse_p_weight)

        return batch_Q_v + batch_P_v

    def get_sparsity(self, get_n_params=False):
        total_params = self._num_item * self._hidden_size
        self.apply_pruning()

        n_params = 0
        for w in [self.sparse_p_weight, self.sparse_q_weight]:
            n_params += torch.nonzero(w).size(0)

        if get_n_params:
            return (1 - n_params / total_params), n_params
        else:
            return 1 - n_params / total_params

    def get_weight(self):
        self.apply_pruning()

        device = self.sparse_p_weight.device
        all_idxes = torch.arange(self._num_item, device=device)

        return self(all_idxes)
