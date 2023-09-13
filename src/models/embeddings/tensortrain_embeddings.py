import torch
from loguru import logger

from .base import IEmbedding
from .tt_embedding_ops import TT_EMB_AVAILABLE, TTEmbeddingBag


class TTEmbedding(IEmbedding):
    """Wrapper for TTEmbeddingBag"""

    def __init__(
        self,
        num_item,
        hidden_size: int,
        mode=None,
        **kwargs,
    ):
        assert mode in ["mean", "sum", None], f"{mode} is not supported"
        assert TT_EMB_AVAILABLE, "TT Emb is not available"

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

        results = self._tt_emb(x, offsets, warmup=warmup)

        if is_flatten and self._mode is None:
            results = results.reshape(b, n, self._hidden_size)
        return results
