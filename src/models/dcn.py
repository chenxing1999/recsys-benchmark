from typing import Any, Dict, List, Optional, Union, cast

import torch
from loguru import logger
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

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Dict[str, Any]],
        strict=True,
        *,
        empty_embedding=False,
    ):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")

        checkpoint = cast(Dict[str, Any], checkpoint)
        model_config = checkpoint["model_config"]
        field_dims = checkpoint["field_dims"]

        compile_model = True
        if "compile_model" in model_config:
            compile_model = model_config.pop("compile_model")

        model = cls(field_dims, **model_config, empty_embedding=empty_embedding)

        if compile_model:
            model = torch.compile(model)

        missing, unexpected = model.load_state_dict(
            checkpoint["state_dict"], strict=strict
        )
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        return model
