from typing import Dict, List, Optional

import torch
from torch import nn

from .embeddings import IEmbedding, get_embedding
from .layer_dcn import DCN_MixHead


class DCN_Mix(nn.Module):
    embedding: IEmbedding

    def __init__(
        self,
        field_dims: List[int],
        num_factor: int,
        hidden_sizes: List[int],
        num_layers: int = 3,
        num_experts: int = 4,
        rank: int = 64,
        activation: Optional[str] = None,
        embedding_config: Optional[Dict] = None,
        p_dropout=0.5,
        empty_embedding=False,
    ):
        """
        DCN Mixture with Stacked Structure

        Args:
            field_dims: List of field dimension for each features
            num_factor: Low-level embedding vector dimension
        """
        super().__init__()

        if not embedding_config:
            embedding_config = {"name": "vanilla"}

        sum(field_dims)

        if not empty_embedding:
            self.embedding = get_embedding(
                embedding_config,
                field_dims,
                num_factor,
                mode=None,
                field_name="dcn",
            )

        inp_size = num_factor * len(field_dims)
        self.cross_head = DCN_MixHead(
            num_experts,
            num_layers,
            rank,
            inp_size,
            activation,
        )
        layers: List[nn.Module] = []
        for size in hidden_sizes:
            layers.append(nn.Linear(inp_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            inp_size = size

        layers.append(nn.Linear(inp_size, 1))
        self._dnn = nn.Sequential(*layers)

        field_dims_tensor = torch.tensor(field_dims)
        field_dims_tensor = torch.cat(
            [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        )

        offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
        self.register_buffer("offsets", offsets)

    def forward(self, x):
        """
        Args:
            x: torch.LongTensor (Batch x NumField)

        Return:
            scores: torch.FloatTensor (Batch): Logit result before sigmoid
        """
        x = x + self.offsets

        # start with embedding layer
        emb = self.embedding(x)

        # followed by a cross net
        bs = x.shape[0]
        emb = emb.view(bs, -1)
        cross_logit = self.cross_head(emb)

        scores = self._dnn(cross_logit)
        scores = scores.squeeze(-1)
        return scores
