from typing import List, Optional

import numpy as np
import torch
from einops import einsum, rearrange
from loguru import logger
from torch import nn

from .base import IEmbedding
from .tt_embedding_ops import (
    TT_EMB_AVAILABLE,
    TTEmbeddingBag,
    get_num_params,
    reset_parameters,
    suggested_tt_shapes,
    tt_matrix_to_full,
)


class TTEmbedding(IEmbedding):
    """Wrapper for TTEmbeddingBag"""

    def __init__(
        self,
        field_dims,
        hidden_size: int,
        mode=None,
        **kwargs,
    ):
        assert mode in ["mean", "sum", None], f"{mode} is not supported"
        assert TT_EMB_AVAILABLE, "TT Emb is not available"

        if isinstance(field_dims, int):
            field_dims = [field_dims]

        num_item = sum(field_dims)

        super().__init__()

        # Should add default tt_ranks
        self._tt_emb = TTEmbeddingBag(
            num_item,
            hidden_size,
            **kwargs,
        )
        self._mode = mode
        self._num_item = num_item
        self._hidden_size = hidden_size
        self._field_dims = field_dims

        logger.debug(f"P Shapes (num_item dim): {self._tt_emb.tt_p_shapes}")
        logger.debug(f"Q Shapes (hidden dim): {self._tt_emb.tt_q_shapes}")

    def get_weight(self):
        return self._tt_emb.full_weight()[: self._num_item, : self._hidden_size]

    def forward(self, x, warmup=True):
        """
        Args:
            x:
                shape B x N
                shape B
        """
        inp_device = x.device
        device = self._tt_emb.tt_cores[0].data.device
        assert (
            device.type != "cpu" and inp_device.type != "cpu"
        ), "CPU Operation is not supported by TTEmbedding"

        is_flatten = False
        step = 1
        size = x.shape[0]
        if len(x.shape) == 2:
            b, n = x.shape
            size = x.shape[0] * x.shape[1]

            if self._mode is not None:
                step = x.shape[1]

            is_flatten = True
            x = x.flatten()

        offsets = torch.arange(0, size + 1, step=step, device=device)

        results = self._tt_emb(x, offsets)

        if is_flatten and self._mode is None:
            results = results.reshape(b, n, self._hidden_size)
        return results


# --- Python implemenation
def reshape_cores(
    tt_p_shapes: List[int],
    tt_q_shapes: List[int],
    tt_ranks: List[int],
    tt_cores: List[torch.Tensor],
    tt_permute: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    tt_ndim = len(tt_p_shapes)
    if len(tt_ranks) == tt_ndim - 1:
        tt_ranks = [1] + tt_ranks + [1]
    tt_cores_ = []
    if tt_permute is not None:
        for i, t in enumerate(tt_cores):
            size_tt = [tt_ranks[i], tt_p_shapes[i], tt_q_shapes[i], tt_ranks[i + 1]]
            size_tt_permute = [0] * 4
            for i in range(4):
                size_tt_permute[i] = size_tt[tt_permute[i]]
            tt_cores_.append(t.view(*size_tt_permute).permute(*tt_permute).contiguous())
    return tt_cores_


def _core_dot_prod(core1, core2):
    # r= Rank, b=Batch, h=Hidden
    res = einsum(core1, core2, "r0 b h0 j, j b h1 r1 -> r0 b h0 h1 r1")
    res = rearrange(res, "r0 b h0 h1 r1 -> r0 b (h0 h1) r1")
    return res


def tt_rec_torch_forward(
    indices: torch.Tensor,
    tt_p_shapes: List[int],
    tt_q_shapes: List[int],
    tt_ranks: List[int],
    cores: List[torch.Tensor],
) -> torch.Tensor:
    # reshape tt_cores to the shape in paper
    cores = reshape_cores(tt_p_shapes, tt_q_shapes, tt_ranks, cores, [1, 0, 2, 3])
    big_bucket: int = torch.prod(torch.tensor(tt_p_shapes)).item()

    res: torch.Tensor
    for idx, dim in enumerate(tt_p_shapes):
        big_bucket //= dim
        v = indices // big_bucket
        indices = indices % big_bucket

        if idx == 0:
            res = cores[idx][:, v]
        else:
            res = _core_dot_prod(res, cores[idx][:, v])

    return res.squeeze([0, 3])


class TTRecTorch(IEmbedding):
    """Reimplemation of TensorTrain with Pytorch.
    This implementation doesnt support cache and sparse forward."""

    def __init__(
        self,
        field_dims,
        hidden_size: int,
        tt_ranks: List[int],
        mode=None,
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        weight_dist: str = "approx-normal",
        enforce_embedding_dim: bool = False,
    ):
        super().__init__()

        if isinstance(field_dims, int):
            field_dims = [field_dims]
        num_embeddings = sum(field_dims)
        embedding_dim = hidden_size

        # Init tt_p_shapes and tt_q_shapes
        self.tt_p_shapes: List[int] = (
            suggested_tt_shapes(num_embeddings, len(tt_ranks) + 1)
            if tt_p_shapes is None
            else tt_p_shapes
        )
        self.tt_q_shapes: List[int] = (
            # if enforce_embedding_dim=True, we make sure that
            # prod(tt_q_shapes) == embedding_dim by disabling round up
            suggested_tt_shapes(
                embedding_dim,
                len(tt_ranks) + 1,
                allow_round_up=(not enforce_embedding_dim),
            )
            if tt_q_shapes is None
            else tt_q_shapes
        )
        assert len(self.tt_p_shapes) >= 2
        assert len(self.tt_p_shapes) <= 4
        assert len(tt_ranks) + 1 == len(self.tt_p_shapes)
        assert len(self.tt_p_shapes) == len(self.tt_q_shapes)
        assert np.prod(np.array(self.tt_p_shapes)) >= num_embeddings
        assert np.prod(np.array(self.tt_q_shapes)) == embedding_dim
        self.tt_ndim = len(tt_ranks) + 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tt_ranks = [1] + tt_ranks + [1]
        self.num_tables = 1
        logger.debug(
            f"Creating TTRecTorch "
            f"tt_p_shapes: {self.tt_p_shapes}, "
            f"tt_q_shapes: {self.tt_q_shapes}, "
            f"tt_ranks: {self.tt_ranks}, "
        )

        n_params: int = get_num_params(
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
        )
        logger.info(f"Num Params: {n_params}")

        self.tt_cores = nn.ParameterList()
        for i in range(self.tt_ndim):
            self.tt_cores.append(
                torch.nn.Parameter(
                    torch.empty(
                        [
                            self.num_tables,
                            self.tt_p_shapes[i],
                            self.tt_ranks[i]
                            * self.tt_q_shapes[i]
                            * self.tt_ranks[i + 1],
                        ],
                        dtype=torch.float32,
                    )
                )
            )

        # initialize parameters
        reset_parameters(
            self.num_embeddings,
            self.embedding_dim,
            self.tt_ranks,
            weight_dist,
            self.tt_cores,
            self.tt_ndim,
            self.tt_p_shapes,
            self.tt_q_shapes,
        )

    def get_num_params(self):
        return get_num_params(
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
        )

    def get_weight(self):
        return tt_matrix_to_full(
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
            self.tt_cores,
            [1, 0, 2, 3],
        )[: self.num_embeddings, : self.embedding_dim]

    def forward(self, x):
        flatten_ind = x.flatten()
        return tt_rec_torch_forward(
            flatten_ind,
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
            self.tt_cores,
        ).reshape(*x.shape, self.embedding_dim)

    def copy_weight_fbtt_weight(self, emb: TTEmbedding):
        """Copy weight from the original implementation"""
        for core, core_torch in zip(emb._tt_emb.tt_cores, self.tt_cores):
            assert core_torch.data.shape == core.data.shape
            core_torch.data = core.data

    @classmethod
    def init_from_fbtt_weight(cls, emb: TTEmbedding):
        tt_ops = emb._tt_emb
        field_dims = emb._field_dims

        new_emb = cls(
            field_dims,
            emb._hidden_size,
            tt_ops.tt_ranks,
            tt_p_shapes=tt_ops.tt_p_shapes,
            tt_q_shapes=tt_ops.tt_q_shapes,
        )
        new_emb.copy_weight_from_fbtt_weight(emb)
        return new_emb
