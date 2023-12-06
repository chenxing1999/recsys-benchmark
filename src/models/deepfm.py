import os
from typing import Dict, List, Optional, Union

import torch
from loguru import logger
from torch import nn

from .embeddings import IEmbedding, get_embedding


class DeepFM(nn.Module):
    embedding: IEmbedding

    def __init__(
        self,
        field_dims: List[int],
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

        layers: List[nn.Module] = []
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

        field_dims_tensor = torch.tensor(field_dims)
        field_dims_tensor = torch.cat(
            [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        )
        offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
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

    def get_ranks(self, x) -> torch.Tensor:
        scores = self(x)
        return torch.argsort(scores, descending=True)

    @classmethod
    def load(cls, checkpoint: Union[str, Dict], strict=True) -> "DeepFM":
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        model_config = checkpoint["model_config"]
        field_dims = checkpoint["field_dims"]

        model = cls(field_dims, **model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model


def save_model_checkpoint(
    model: DeepFM,
    checkpoint_dir: str,
    name: str = "target",
):
    """Wrapper to save checkpoint embedding to a folder
    in the belowing format:
        {checkpoint_dir}/deepfm/{name}.pth
    """

    emb = model.embedding
    field_dir = os.path.join(checkpoint_dir, "deepfm")
    os.makedirs(field_dir, exist_ok=True)

    path = os.path.join(field_dir, f"{name}.pth")
    torch.save(emb.state_dict(), path)


def get_optimizers(
    model: DeepFM,
    config: Dict,
) -> List[torch.optim.Optimizer]:
    sparse: bool = config.get("sparse", False)
    optimizer_name: str = config.get("optimizer", "adam")

    lr_emb = config.get("learning_rate_emb", config["learning_rate"])

    logger.debug(f"optimizer config: {sparse=} - {optimizer_name=} - {lr_emb=}")

    decay_param = []
    if sparse:
        decay_param = [
            p for name, p in model.named_parameters() if "embedding." not in name
        ]
        no_decay_param = model.embedding.parameters()

    if sparse and optimizer_name == "adam":
        return [
            torch.optim.SparseAdam(
                no_decay_param,
                lr=lr_emb,
            ),
            torch.optim.Adam(
                decay_param,
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            ),
        ]

    if optimizer_name == "adam":
        return [
            torch.optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        ]
    elif optimizer_name == "sgd":
        if not sparse:
            return [
                torch.optim.SGD(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
            ]
        else:
            return [
                torch.optim.SGD(
                    [
                        dict(params=no_decay_param, weight_decay=0, lr=lr_emb),
                        dict(
                            params=decay_param,
                            weight_decay=config["weight_decay"],
                            lr=config["learning_rate"],
                        ),
                    ],
                )
            ]

    else:
        raise ValueError(f"{optimizer_name=} is not recognized")
