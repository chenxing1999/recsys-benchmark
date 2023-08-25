from typing import List, Optional, Union

from torch import nn


class IEmbedding(nn.Module):
    """
    Note: Not all weight is initialized from the same distribution
    as I found out that the Xavier uniform despite converge slower
    but usually get higher accuracy.
    """

    def get_weight(self):
        ...


class VanillaEmbedding(IEmbedding):
    """Wrapper for default PyTorch Embedding / EmbeddingBag"""

    _emb_module: Union[nn.Embedding, nn.EmbeddingBag]

    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str] = None,
        initializer="xavier",
        **kwargs,
    ):
        """
        Args:
            field_dims: List of Field dimension or Single value field size
            hidden_size: Embedding size
            initializer: normal or xavier
            mode: See torch.nn.EmbeddingBag's Documentation for more information
                if None, always use torch.nn.Embedding

            **kwargs: Will be based to _emb_module
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in [None, "sum", "mean", "max"]

        if mode is None:
            self._emb_module = nn.Embedding(
                sum(field_dims),
                hidden_size,
                **kwargs,
            )
        else:
            self._emb_module = nn.EmbeddingBag(
                sum(field_dims),
                hidden_size,
                mode=mode,
                **kwargs,
            )

        if initializer == "xavier":
            nn.init.xavier_uniform_(self._emb_module.weight)
        else:
            nn.init.normal_(self._emb_module.weight, std=0.1)

    def get_weight(self):
        return self._emb_module.weight

    def forward(self, x):
        return self._emb_module(x)
