from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from .embeddings import IEmbedding, get_embedding


class DeepFM(nn.Module):
    embedding: IEmbedding

    def __init__(
        self,
        field_dims: Sequence[int],
        num_factor: int,
        hidden_sizes: List[int],
        p_dropout: float = 0.1,
        use_batchnorm=False,
        embedding_config: Optional[Dict] = None,
    ):
        """
        Args:
            field_dims: List of field dimension for each features
            num_factor: Low-level embedding vector dimension
            hidden_sizes: MLP layers' hidden sizes
            p_dropout: Dropout rate per MLP layer
            embedding_config
        """

        super().__init__()

        if not embedding_config:
            embedding_config = {"name": "vanilla"}

        num_inputs = sum(field_dims)
        self.embedding = get_embedding(
            embedding_config,
            field_dims,
            num_factor,
            mode=None,
            field_name="deepfm",
        )

        self.fc = nn.EmbeddingBag(num_inputs, 1, mode="sum")
        self.linear_layer = nn.Linear(1, 1)
        self._bias = nn.Parameter(torch.zeros(1))

        deep_branch_inp = num_factor * len(field_dims)

        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(deep_branch_inp, size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            deep_branch_inp = size

        layers.append(nn.Linear(deep_branch_inp, 1))
        self._deep_branch = nn.Sequential(*layers)

        # torch.set_float32_matmul_precision('high')
        # self._deep_branch = torch.compile(self._deep_branch)

        field_dims = torch.tensor(field_dims)
        field_dims = torch.cat([torch.tensor([0], dtype=torch.long), field_dims])
        offsets = torch.cumsum(field_dims[:-1], 0).unsqueeze(0)
        self.register_buffer("offsets", offsets)

    # @torch.compile(fullgraph=True)
    def forward(self, x):
        """
        Args:
            x: torch.LongTensor (Batch x NumField)

        Return:
            scores: torch.FloatTensor (Batch): Logit result before sigmoid
        """

        x = x + self.offsets
        emb = self.embedding(x)

        square_of_sum = emb.sum(dim=1).pow(2)
        sum_of_square = emb.pow(2).sum(dim=1)

        # x_1 = alpha * WX + b
        x = self.fc(x) + self._bias
        # x = self.fc(x) + self._bias
        # x_2 = alpha * WX + b + 0.5 ((\sum e_i)^2 - (\sum e_i^2))
        y_fm = x + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)

        b, num_field, size = emb.shape
        emb = emb.reshape((b, num_field * size))
        scores = y_fm + self._deep_branch(emb)
        scores = scores.squeeze(-1)

        return scores

    @classmethod
    def load(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint["model_config"]
        field_dims = checkpoint["field_dims"]

        model = cls(field_dims, **model_config)
        return model
