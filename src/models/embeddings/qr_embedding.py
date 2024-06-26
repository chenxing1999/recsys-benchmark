import math
from typing import List, Literal, Optional, Union

import torch
from torch import nn

from .base import IEmbedding


class QRHashingEmbedding(IEmbedding):
    """QUOTIENT-REMAINDER based hasing Embedding
    Original paper: https://arxiv.org/pdf/1909.02107.pdf
    """

    def __init__(
        self,
        field_dims: Union[int, List[int]],
        hidden_size: int,
        mode: Optional[str] = None,
        divider: Optional[int] = None,
        operation: Literal["cat", "add", "mult"] = "mult",
        initializer="uniform",
    ):
        """
        Args:
            field_dims: Categorical feature sizes
            hidden_size: Output embedding hidden size.
                For simplicity, we dont allow different size for concatenate feature

            divider: Hash collision mentioned in the paper
            operation: Support:
                cat (concatenate): Concatenating feature
                add: Adding feature to each other
                mult: Element wise multiplication
        """

        super().__init__()
        assert operation in ["cat", "add", "mult"]
        if operation == "cat":
            assert hidden_size % 2 == 0

        if isinstance(field_dims, int):
            field_dims: List[int] = [field_dims]

        num_item = sum(field_dims)

        if divider is None:
            divider = int(math.sqrt(num_item))

        emb_size = hidden_size
        if operation == "cat":
            emb_size = hidden_size // 2

        self._operation = operation

        size = (num_item - 1) // divider + 1

        if mode is None:
            self.emb1 = nn.Embedding(divider, emb_size)
            self.emb2 = nn.Embedding(size, emb_size)
        else:
            self.emb1 = nn.EmbeddingBag(divider, emb_size, mode=mode)
            self.emb2 = nn.EmbeddingBag(size, emb_size, mode=mode)

        self._hidden_size = hidden_size
        self._divider = divider
        self._num_item = num_item

        if initializer == "normal":
            self._init_normal_weight()
        elif initializer == "uniform":
            self._init_uniform_weight()

    def _init_uniform_weight(self):
        # Haven't found method to generate emb1 and emb2
        # so that ops(emb1, emb2) will have uniform(-a, a).
        # Note: Orignal QR init method use uniform(sqrt(1 / num_categories), 1)
        # https://github.com/facebookresearch/dlrm/blob/dee060be6c2f4a89644acbff6b3a36f7c0d5ce39/tricks/qr_embedding_bag.py#L182-L183
        # I use similiar distribution, I think that the reason for this
        # distribution is when multiply, there would be no zero multiply with zero

        alpha = math.sqrt(1 / self._num_item)
        nn.init.uniform_(self.emb1.weight, alpha, 1)
        nn.init.uniform_(self.emb2.weight, alpha, 1)

    def _init_normal_weight(self):
        std = 0.1
        if self._operation == "add":
            std = std / 2
        elif self._operation == "mult":
            std = math.sqrt(std)
        nn.init.normal_(self.emb1.weight, std=std)
        nn.init.normal_(self.emb2.weight, std=std)

    def forward(self, tensor: torch.IntTensor):
        inp1 = tensor % self._divider
        inp2 = tensor // self._divider

        emb1 = self.emb1(inp1)
        emb2 = self.emb2(inp2)

        if self._operation == "cat":
            return torch.cat([emb1, emb2], dim=1)
        elif self._operation == "add":
            return emb1 + emb2
        elif self._operation == "mult":
            return emb1 * emb2
        else:
            raise NotImplementedError("Unsupported operation: {self._operation}")

    def get_weight(self):
        arr = torch.arange(self._num_item, device=self.emb1.weight.device)
        return self(arr)
