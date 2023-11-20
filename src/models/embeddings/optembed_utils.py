from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
from torch import nn


def get_mask(hidden_size: int):
    """
    Return a matrix tensor
        matrix[i][j] = 1 if i >= j
    """
    # matrix size  = hidden_size * hidden_size
    # Number of ones = \sum i with i from 1 to hidden_size + 1
    # then drop it

    values = torch.ones((hidden_size, hidden_size), dtype=torch.bool)
    matrix = torch.tril(values)

    return matrix


class BinaryStep(torch.autograd.Function):
    """Copied from original OptEmbed repo
    (https://github.com/fuyuanlyu/OptEmbed)
    """

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return (inp > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(inp) > 1
        middle_index = (torch.abs(inp) <= 1) * (torch.abs(inp) > 0.4)
        additional = 2 - 4 * torch.abs(inp)
        additional[zero_index] = 0.0
        additional[middle_index] = 0.4
        return grad_input * additional


class _MaskEmbeddingModule(nn.Module):
    """Wrapper to apply Mask embedding"""

    def __init__(
        self,
        field_dims: torch.LongTensor,
        t_init: float = 0,
        mode_threshold_e="field",
        norm=1,
    ):
        super().__init__()
        self.s = BinaryStep.apply
        assert mode_threshold_e in ["feature", "field"]
        self.mode_threshold_e = mode_threshold_e

        self.register_buffer("_field_dims", field_dims)

        self._num_item = self._field_dims.sum()
        self._num_field = len(field_dims)

        if self.mode_threshold_e == "feature":
            t_size = self._num_item
        else:
            t_size = self._num_field

        self._t_param = nn.Parameter(torch.empty(t_size))
        nn.init.constant_(self._t_param, t_init)
        self._norm = norm

    def _transform_t_to_feat(self):
        """Transform vector t from current format to num feature-dimension"""
        if self.mode_threshold_e == "feature":
            return self._t_param

        return torch.repeat_interleave(
            self._t_param,
            self._field_dims,
            dim=0,
            output_size=self._num_item,
        )

    def forward(self, x):
        """
        x: shape B x N x D or shape F x D
            if shape Batch x Num field x Dimension: Apply mask to second dimension
            if shape Num feature x Dimension:
        """
        if len(x.shape) == 2:
            weight = x
            t = self._transform_t_to_feat()
            mask_e = self.s(torch.norm(weight, self._norm, dim=1) - t)
            mask_e = mask_e.unsqueeze(-1)
            emb = weight * mask_e
        else:
            assert self.mode_threshold_e == "field", "Cannot apply field mask to input"
            mask = self.s(torch.norm(x, self._norm, dim=2) - self._t_param)
            mask = mask.unsqueeze(-1)
            emb = x * mask

        return emb


@lru_cache(1)
def _find_alpha(
    target_sparsity,
    hidden_size,
    step=0.1,
    eps=1e-6,
    num_step=100,
):
    """Find alpha based on target sparsity and hidden size using gradient descend"""
    if target_sparsity == 0.7 and hidden_size == 64:
        return 1.045
    elif target_sparsity == 0.8 and hidden_size == 64:
        return 1.083
    elif target_sparsity == 0.5:
        return 1

    # brute force find alpha with gradient descend :))))
    if target_sparsity > 0.5:
        alpha = torch.tensor(1.1, requires_grad=True)
    else:
        alpha = torch.tensor(0.9, requires_grad=True)
    for i in range(num_step):
        expected_hidden_size = _get_expected_hidden_size(alpha, hidden_size)
        expected_sparsity = 1 - expected_hidden_size / hidden_size

        # To increase sparsity -> increase alpha
        diff = expected_sparsity - target_sparsity
        if abs(diff) < eps and diff > 0:
            return alpha.item()

        alpha.grad = None
        diff = diff**2
        diff.backward()
        alpha.data -= step * alpha.grad

    return alpha.item()


def _get_expected_hidden_size(
    alpha: Union[float, torch.Tensor], max_hidden_size
) -> Union[float, torch.Tensor]:
    """Quick function to calculate expected hidden size sampling from
    p_i = alpha^(h - i), with p_i is prob of sample i hidden size
    """
    if alpha == 1:
        return (max_hidden_size + 1) / 2
    return alpha / (alpha - 1) - max_hidden_size / (alpha**max_hidden_size - 1)


def _generate_weight(alpha, hidden_size):
    f = np.power(alpha, np.arange(1, hidden_size + 1) * (-1) + hidden_size)
    p = f / f.sum()
    return p


def _get_linear_hidden(target_sparsity, hidden_size):
    assert (
        target_sparsity >= 0.5
    ), "Generate naive only could generate sparsity from 0.5"
    hidden_size = int(hidden_size * 2 * (1 - target_sparsity))
    return hidden_size


def _sampling_by_weight(
    target_sparsity: Optional[int],
    hidden_size: int,
    num_item: int,
    method=0,
    device=None,
):
    """Sampling mask d based on given input

    Args:
        target_sparsity: If None, just generate mask based on
            uniform distribution

    """
    if target_sparsity is None:
        return torch.randint(0, hidden_size, (num_item,), device=device)

    if method == 2:
        hidden = _get_linear_hidden(target_sparsity, hidden_size)
        return torch.randint(0, hidden, (num_item,), device=device)

    alpha = _find_alpha(target_sparsity, hidden_size)
    weight = _generate_weight(alpha, hidden_size)
    sampler = torch.utils.data.WeightedRandomSampler(weight, num_item)
    return torch.tensor(list(sampler), device=device)
