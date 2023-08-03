from typing import List, Sequence

from torch import nn


class DeepFM(nn.Module):
    def __init__(
        self,
        field_dims: Sequence[int],
        num_factor: int,
        hidden_sizes: List[int],
        p_dropout: float = 0.1,
    ):
        """
        Args:
            field_dims: List of field dimension for each features
            num_factor: Low-level embedding vector dimension
            hidden_sizes: MLP layers' hidden sizes
            p_dropout: Dropout rate per MLP layer
        """

        super().__init__()
        num_inputs = int(sum(field_dims))

        self.embedding = nn.Embedding(num_inputs, num_factor)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Linear(1, 1)

        deep_branch_inp = num_factor * len(field_dims)
        p_dropout = 0.1

        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(deep_branch_inp, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            deep_branch_inp = size

        layers.append(nn.Linear(deep_branch_inp, 1))
        self._deep_branch = nn.Sequential(*layers)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        emb = self.embedding(x)

        square_of_sum = emb.sum(dim=1).pow(2)
        sum_of_square = emb.pow(2).sum(dim=1)

        x = self.linear_layer(self.fc(x).sum(1))
        x = x + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)

        b, num_field, size = emb.shape
        emb = emb.reshape((b, num_field * size))
        x = x + self._deep_branch(emb)

        return x
