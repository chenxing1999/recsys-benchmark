"""Quantization Aware Training Embedding for Recommender"""
from typing import List, Tuple, Union

import torch
from torch import nn

from src.models.embeddings.base import VanillaEmbedding


def get_qmax_qmin(n_bits: torch.Tensor) -> Tuple[float, float]:
    q_min = (-1) * (1 << (n_bits - 1))
    q_max = (1 << (n_bits - 1)) - 1
    return q_max.item(), q_min.item()


class StotasticRounding(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(ctx, scale, w, n_bits):
        """
        Args:
            scale: scalar value
            w: (2 or 3) dimension FloatTensor
            n_bits: int
        Returns:
            q_w: (2 or 3) IntTensor
        """

        q_w_float = w / scale

        q_max, q_min = get_qmax_qmin(n_bits)

        # Clip(w / scale)
        q_w = torch.clamp(q_w_float, q_min, q_max)

        # Stotastic Rounding
        q_w_floor = torch.floor(q_w)
        prob_floor = q_w_floor + 1 - q_w
        prob = torch.rand_like(prob_floor)

        is_ceil = prob > prob_floor
        q_w = q_w_floor + is_ceil
        ctx.save_for_backward(q_w, q_w_float, n_bits)
        return q_w * scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: 2 or 3D array
        Returns:
            grad_scale: 1 scalar value
            grad_w: 2 or 3D array
            grad_n_bits: 0
        """
        (res, q_w, n_bits) = ctx.saved_tensors

        q_max, q_min = get_qmax_qmin(n_bits)
        scale_mask = torch.empty_like(grad_output)

        # TODO: This have not been finished
        # implement scale_mask
        mask1 = q_w >= q_max
        mask2 = q_w <= q_min
        scale_mask[mask1] = q_max
        scale_mask[mask2] = q_min

        mask3 = torch.logical_and(~mask1, ~mask2)

        # arr = q_w[mask3]
        # print(q_w, res)

        # (P.Round()(q_w)-q_w)
        # q_w=q_w, res=P.Round()(q_w)
        arr = q_w[mask3] - res[mask3]
        scale_mask[mask3] = -arr
        # scale_mask = q_w - res

        # 3, 5, 7
        grad_scale = grad_output * scale_mask

        return grad_scale, grad_output.clone(), torch.tensor(0)


class QAT_EmbInt(VanillaEmbedding):
    def __init__(
        self,
        field_dims: Union[int, List[int]],
        num_factor: int = 16,
        mode=None,
        initializer="xavier",
        *,
        stochastic_rounding: bool = True,
        n_bits=8,
        fixed_scale=False,
        **kwargs,
    ):
        super().__init__(field_dims, num_factor, mode, initializer, **kwargs)
        assert n_bits in [8, 16]

        assert (
            stochastic_rounding
        ), "Not implement deterministic yet, I am lazy, please wait"
        self.n_bits = torch.tensor(n_bits)

        if isinstance(field_dims, int):
            field_dims = [field_dims]

        q_max, q_min = get_qmax_qmin(self.n_bits)

        with torch.no_grad():
            w = super().get_weight()
            scale_init = (w.max() - w.min()) / (q_max - q_min)
        self.register_parameter("scale", nn.Parameter(scale_init, not fixed_scale))

    def forward(self, x):
        emb = super(QAT_EmbInt, self).forward(x)
        return StotasticRounding.apply(self.scale, emb, self.n_bits)

    def get_weight(self):
        return super().get_weight()
