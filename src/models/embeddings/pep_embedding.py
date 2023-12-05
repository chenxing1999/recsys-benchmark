import os
from typing import List, Optional, Union

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from .base import IEmbedding


class PepEmbeeding(IEmbedding):
    """PEP Embedding proposed in
        https://arxiv.org/pdf/2101.07577.pdf

    Original implementation:
        https://github.com/ssui-liu/learnable-embed-sizes-for-RecSys
    """

    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str] = None,
        ori_weight_dir: str = "",
        checkpoint_weight_dir: str = "checkpoints",
        field_name: str = "",
        init_threshold: float = -150,
        threshold_type: str = "feature_dim",
        sparsity: Optional[List[float]] = None,
    ):
        """
        Args:
            field_dims
            hidden_size
            ori_weight_path: Path to original weight for later retrain
                with Lottery Ticket. If not given, do nothing
            checkpoint_weight_dir: Path to save checkpoint. Checkpoint full path
                will be {checkpoint_weight_dir}/{field_name}/{sparsity}.pth)
            init_threshold: Initialize value for s
            threshold_type: Support `feature_dim`, `feature`, `dimension` and `global
            sparsity
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        num_item = sum(field_dims)

        if sparsity is None:
            sparsity = [0.8, 0.9, 0.99]

        assert isinstance(sparsity, list) and isinstance(sparsity[0], float)
        self.sparsity = list(sorted(sparsity))
        self._cur_min_spar_idx = 0

        # Initialize and Save the original weight to get winning ticket later
        self.emb = nn.Embedding(num_item, hidden_size)
        nn.init.xavier_uniform_(self.emb.weight)

        if ori_weight_dir:
            logger.debug("ori_weight_dir given. Saving initialize checkpoint...")
            os.makedirs(ori_weight_dir, exist_ok=True)
            ori_weight_path = os.path.join(ori_weight_dir, field_name + ".pth")
            torch.save({"state_dict": self.emb.state_dict()}, ori_weight_path)

        self.threshold_type = threshold_type
        self.s = self.init_threshold(init_threshold, num_item, hidden_size)

        self.field_name = field_name

        if field_name:
            checkpoint_weight_dir = os.path.join(checkpoint_weight_dir, field_name)
        os.makedirs(checkpoint_weight_dir, exist_ok=True)
        self.checkpoint_weight_dir = checkpoint_weight_dir
        self._mode = mode

    def get_weight(self):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        return sparse_weight

    def forward(self, x):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        if self._mode:
            xv = F.embedding_bag(x, sparse_weight, mode=self._mode)
        else:
            xv = F.embedding(x, sparse_weight)

        return xv

    def soft_threshold(self, v, s):
        return torch.sign(v) * torch.relu(torch.abs(v) - torch.sigmoid(s))

    def init_threshold(self, init, num_item, hidden_size) -> nn.Parameter:
        """

        threshold_type
            global: single threshold for all item
            dimension: threshold for all dimension


        """
        if self.threshold_type == "global":
            s = nn.Parameter(init * torch.ones(1))
        elif self.threshold_type == "dimension":
            s = nn.Parameter(init * torch.ones([hidden_size]))
        elif self.threshold_type == "feature":
            s = nn.Parameter(init * torch.ones([num_item, 1]))
        elif self.threshold_type == "field":
            raise NotImplementedError()
        elif self.threshold_type == "feature_dim":
            s = nn.Parameter(init * torch.ones([num_item, hidden_size]))
        elif self.threshold_type == "field_dim":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid threshold_type: {}".format(self.threshold_type))
        return s

    def get_sparsity(self, get_n_params=False) -> float:
        total_params = self.emb.weight.numel()
        n_params = self.get_num_params()
        if get_n_params:
            return (1 - n_params / total_params), n_params
        else:
            return 1 - n_params / total_params

    def get_num_params(self) -> int:
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        n_params = torch.count_nonzero(sparse_weight).item()
        return n_params

    def train_callback(self):
        """Callback to save weight to `checkpoint_weight_dir`"""
        with torch.no_grad():
            cur_sparsity = self.get_sparsity()

        while (
            self._cur_min_spar_idx < len(self.sparsity)
            and self.sparsity[self._cur_min_spar_idx] < cur_sparsity
        ):
            sparsity = self.sparsity[self._cur_min_spar_idx]
            logger.info(f"cur_sparsity is larger than {sparsity}")

            # Save model
            path = os.path.join(self.checkpoint_weight_dir, f"{sparsity}.pth")
            torch.save(self.state_dict(), path)
            self._cur_min_spar_idx += 1


class RetrainPepEmbedding(IEmbedding):
    """Wrapper for Retrain inference logic of Pep Embedding"""

    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size,
        mode: Optional[str],
        checkpoint_weight_dir,
        sparsity: Union[float, str] = 0.8,
        ori_weight_dir: Optional[str] = None,
        field_name: str = "",
        sparse=False,
    ):
        """
        Args:
            field_dims
            hidden_size
            checkpoint_weight_dir: Path pep_checkpoint folder
                The weight to get weight mask from should be
                {checkpoint_weight_dir}/{field_name}/{sparsity}.pth

            sparsity: Target sparsity / name of checkpoint in folder
            ori_weight_dir: Path to original weight
                if not provided, you could try to manually load the original weight
            field_name: Name to field (used to get checkpoint mask path)
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        num_item = sum(field_dims)

        self.emb = nn.Embedding(num_item, hidden_size)

        if ori_weight_dir:
            ori_weight_path = os.path.join(ori_weight_dir, field_name + ".pth")
            original_state_dict = torch.load(ori_weight_path, map_location="cpu")[
                "state_dict"
            ]
            self.emb.load_state_dict(original_state_dict)

        finish_weight_path = os.path.join(
            checkpoint_weight_dir,
            field_name,
            f"{sparsity}.pth",
        )
        finish_weight = torch.load(finish_weight_path, map_location="cpu")
        weight = finish_weight["emb.weight"]
        s = finish_weight["s"]

        # convert to nn.Parameter so that when call .to
        # mask will be moved to correct device
        self.mask = nn.Parameter((torch.abs(weight) - torch.sigmoid(s)) > 0, False)
        nnz = self.mask.sum()
        self._nnz = nnz
        self.sparsity = 1 - (nnz / torch.prod(torch.tensor(self.mask.size()))).item()
        self._mode = mode

        self._sparse = sparse

    def get_weight(self):
        sparse_emb = self.emb.weight * self.mask
        return sparse_emb

    def forward(self, x):
        sparse_emb = self.emb.weight * self.mask
        if self._mode:
            xv = F.embedding_bag(x, sparse_emb, mode=self._mode, sparse=self._sparse)
        else:
            xv = F.embedding(x, sparse_emb, sparse=self._sparse)
        return xv

    def get_sparsity(self, get_n_params=False):
        if get_n_params:
            return self.sparsity, self._nnz
        return self.sparsity

    def get_num_params(self):
        return self._nnz
