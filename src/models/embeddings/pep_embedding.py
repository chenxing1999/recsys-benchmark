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
        num_item: int,
        hidden_size: int,
        ori_weight_dir: str,
        checkpoint_weight_dir: str,
        field_name: str = "",
        init_threshold: float = -150,
        threshold_type: str = "feature_dim",
        sparsity: Optional[List[float]] = None,
    ):
        """
        Args:
            num_item
            hidden_size
            ori_weight_path: Path to original weight for later retrain
                with Lottery Ticket
            init_threshold: Initialize value for s
            threshold_type: Support `feature_dim`, `feature`, `dimension` and `global
            sparsity
        """
        super().__init__()

        if sparsity is None:
            sparsity = [0.8, 0.9, 0.99]
        self.sparsity = list(sorted(sparsity))
        self._cur_min_spar_idx = 0

        # Initialize and Save the original weight to get winning ticket later
        self.emb = nn.Embedding(num_item, hidden_size)
        nn.init.xavier_uniform_(self.emb.weight)

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

    def get_weight(self):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        return sparse_weight

    def forward(self, x):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
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

    def get_sparsity(self) -> float:
        total_params = self.emb.weight.numel()
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        n_params = (sparse_weight != 0).sum()
        return (1 - n_params / total_params).item()

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
        num_item: int,
        hidden_size,
        checkpoint_weight_dir,
        sparsity: Union[float, str],
        ori_weight_dir: Optional[str] = None,
        field_name: str = "",
    ):
        """
        Args:
            num_item
            hidden_size
            checkpoint_weight_dir: Path pep_checkpoint folder
                The weight to get weight mask from should be
                    f"{checkpoint_weight_dir}/{field_name}/{sparsity}.pth

            sparsity: Target sparsity
            ori_weight_dir: Path to original weight
                if not provided, you could try to manually load the original weight
            field_name: Name to field (used to get checkpoint mask path)
        """
        super().__init__()
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
        self.mask = (torch.abs(weight) - torch.sigmoid(s)) > 0
        self.mask = self.mask.cuda()

        nnz = self.mask.sum()
        self.sparsity = 1 - (nnz / torch.prod(torch.tensor(self.mask.size()))).item()

    def get_weight(self):
        sparse_emb = self.emb.weight * self.mask
        return sparse_emb

    def forward(self, x):
        sparse_emb = self.emb.weight * self.mask
        return F.embedding(x, sparse_emb)

    def get_sparsity(self):
        return self.sparsity
