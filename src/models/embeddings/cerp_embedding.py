"""Copy and edited from the original CERP code"""
import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
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
        field_name: str = "",
    ):
        """
        Args:


        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in [None, "sum", "mean", "max"]

        num_item = sum(field_dims)

        self._field_dims = torch.tensor(field_dims)
        self._mode = mode
        self.field_name: str = field_name

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
        """Init threshold wrapper

        Args:
            threshold_type
            init: Initial value for threshold

        Returns:
            if threshold_type == "global":
                return nn.Parameter with shape 1
            elif threshold_type == "element-wise":
                return nn.Parameter with shape row_size * col_size
        """
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
        if self._mode is None:
            batch_Q_v = F.embedding(q_idx, self.sparse_q_weight)
            batch_P_v = F.embedding(p_idx, self.sparse_p_weight)

            emb = batch_Q_v + batch_P_v
            return emb
        elif self._mode in ["sum", "mean"]:
            batch_Q_v = F.embedding_bag(q_idx, self.sparse_q_weight, mode=self._mode)
            batch_P_v = F.embedding_bag(p_idx, self.sparse_p_weight, mode=self._mode)

            emb = batch_Q_v + batch_P_v
            return emb
        else:
            # mode == "max"
            batch_Q_v = F.embedding(q_idx, self.sparse_q_weight)
            batch_P_v = F.embedding(p_idx, self.sparse_p_weight)

            emb = batch_Q_v + batch_P_v
            return emb.max(dim=1)[0]

    def get_sparsity(self, get_n_params=False):
        total_params = self._num_item * self._hidden_size
        self.apply_pruning()

        n_params = 0
        for w in [self.sparse_p_weight, self.sparse_q_weight]:
            n_params += torch.count_nonzero(w).item()

        if get_n_params:
            return (1 - n_params / total_params), n_params
        else:
            return 1 - n_params / total_params

    def get_weight(self):
        device = self.p_weight.data.device
        all_idxes = torch.arange(self._num_item, device=device)

        return self(all_idxes)

    def get_num_params(self):
        self.apply_pruning()

        n_params = 0
        for w in [self.sparse_p_weight, self.sparse_q_weight]:
            n_params += torch.count_nonzero(w).item()
        return n_params

    def get_prune_loss(self, K=100):
        emb = self.sparse_p_weight + self.sparse_q_weight
        return -torch.tanh(emb * K).norm(2) ** 2


class RetrainCerpEmbedding(IEmbedding):
    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str],
        checkpoint_weight_dir: str,
        field_name: str = "",
        weight_name: str = "target",
        bucket_size: int = 8000,
        sparse: bool = False,
    ):
        """
        Args:
            field_dims
            hidden_size
            mode
            checkpoint_weight_dir: Used to load trained model
            field_name: Used to load the original checkpoint.
            weight_name
            bucket_size
            sparse: Required to use with SparseAdam. See torch.Embedding documentation
                for more information

        Note:
            weight will be loaded at
                {checkpoint_weight_dir}/{field_name}/{weight_name}.pth
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        # init weight
        mask_weight_path = os.path.join(
            checkpoint_weight_dir,
            field_name,
            f"{weight_name}.pth",
        )
        init_weight_path = os.path.join(
            checkpoint_weight_dir,
            field_name,
            "initial.pth",
        )

        assert os.path.exists(
            mask_weight_path
        ), f"Weight not found at {mask_weight_path} to re-init mask"
        assert os.path.exists(
            init_weight_path
        ), f"Weight not found at {init_weight_path} to re-init original weight"

        num_item = sum(field_dims)

        self._field_dims = torch.tensor(field_dims)
        self._mode = mode
        self.field_name: str = field_name
        self._bucket_size = bucket_size
        self._hidden_size = hidden_size

        self.p_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )
        self.q_weight = nn.Parameter(
            torch.zeros(bucket_size, hidden_size),
        )
        init_weight = torch.load(init_weight_path)

        init_correct = False
        try:
            assert init_weight["q_weight"].shape == (bucket_size, hidden_size)
            assert init_weight["p_weight"].shape == (bucket_size, hidden_size)
            init_correct = True
        except AssertionError:
            logger.warning("Init weight is not correct. This is expected for inference")

        if init_correct:
            self.q_weight.data = init_weight["q_weight"]
            self.p_weight.data = init_weight["p_weight"]
            self.q_mask, self.p_mask = self.load_mask(mask_weight_path)
        else:
            self.q_mask, self.p_mask = nn.Parameter(
                torch.zeros(self._bucket_size, self._hidden_size)
            ), nn.Parameter(torch.zeros(self._bucket_size, self._hidden_size))

        self._num_item = num_item
        self._hidden_size = hidden_size

        # Q's avg entities per row = \ceiling(#entities / bucket size)
        self.q_entity_per_row = int(np.ceil(self._num_item / self._bucket_size))
        logger.debug(f"{self.q_entity_per_row=}")
        self._sparse = sparse

        self.sparse_p_weight = None
        self.sparse_q_weight = None

    def load_mask(self, weight_path: str) -> List[nn.Parameter]:
        checkpoint = torch.load(weight_path, map_location="cpu")

        names: Tuple[Tuple[str, str], Tuple[str, str]] = (
            ("q_weight", "q_threshold"),
            ("p_weight", "p_threshold"),
        )
        masks = []
        for weight_name, threshold_name in names:
            weight = checkpoint[weight_name]
            threshold = checkpoint[threshold_name]

            mask = (weight.abs() - torch.sigmoid(threshold)) > 0
            assert mask.shape == (self._bucket_size, self._hidden_size)
            masks.append(nn.Parameter(mask, False))
        return masks

    def get_weight(self):
        device = self.p_weight.data.device
        all_idxes = torch.arange(self._num_item, device=device)

        return self(all_idxes)

    def forward(self, x):
        q_idx = torch.div(x, self.q_entity_per_row, rounding_mode="trunc")
        p_idx = x % self._bucket_size
        if self.sparse_q_weight is None or self.training:
            self.sparse_q_weight = self.q_weight * self.q_mask
            self.sparse_p_weight = self.p_weight * self.p_mask

        sparse_q_weight = self.sparse_q_weight
        sparse_p_weight = self.sparse_p_weight

        # get user, items' corresponding embedding vectors in Q, R matrices
        if self._mode is None:
            batch_Q_v = F.embedding(q_idx, sparse_q_weight, sparse=self._sparse)
            batch_P_v = F.embedding(p_idx, sparse_p_weight, sparse=self._sparse)

            emb = batch_Q_v + batch_P_v
            return emb
        elif self._mode in ["sum", "mean"]:
            batch_Q_v = F.embedding_bag(
                q_idx, sparse_q_weight, mode=self._mode, sparse=self._sparse
            )
            batch_P_v = F.embedding_bag(
                p_idx, sparse_p_weight, mode=self._mode, sparse=self._sparse
            )

            emb = batch_Q_v + batch_P_v
            return emb
        else:
            # mode == "max"
            batch_Q_v = F.embedding(q_idx, sparse_q_weight, sparse=self._sparse)
            batch_P_v = F.embedding(p_idx, sparse_p_weight, sparse=self._sparse)

            emb = batch_Q_v + batch_P_v
            return emb.max(dim=1)[0]

    def get_num_params(self):
        if self.sparse_p_weight is None:
            return (
                torch.count_nonzero(self.q_mask) + torch.count_nonzero(self.p_mask)
            ).item()

        return (
            torch.count_nonzero(self.sparse_p_weight)
            + torch.count_nonzero(self.sparse_q_weight)
        ).item()
