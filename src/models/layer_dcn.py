from typing import Literal, Optional

import torch
from einops import einsum
from torch import nn


def forward_mixture(x, C, U, V, act):
    """

    Args:
        x: batch x dim
        C: num_experts x rank x rank
        V: num_experts x dim x rank
        U: num_experts x rank x dim

    Returns:
        E_i: batch x num_experts x dim
    """
    E_i = act(x @ V)
    E_i = E_i.permute(1, 0, 2)
    E_i = act(einsum(E_i, C, "b e r, e r r2 -> b e r2"))
    E_i = einsum(E_i, U, "b e r, e r r2 -> b e r2")
    return E_i


class DCN_MixHead(nn.Module):
    """DeepCross Mixture model"""

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        rank: int,
        hidden_size: int,
        activation: Optional[str] = None,
        gate_act: Literal["softmax", "identity"] = "identity",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.rank = rank

        assert gate_act in ["softmax", "identity"]

        self.U = nn.ParameterList(
            [
                self._init_parameters((num_experts, rank, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.C = nn.ParameterList(
            [
                self._init_parameters((num_experts, rank, rank))
                for _ in range(num_layers)
            ]
        )
        self.V = nn.ParameterList(
            [
                self._init_parameters((num_experts, hidden_size, rank))
                for _ in range(num_layers)
            ]
        )

        self.biases = nn.ParameterList(
            [
                self._init_parameters((1, hidden_size), "zeros")
                for _ in range(num_layers)
            ]
        )

        self.gates = self._init_parameters((num_experts, hidden_size, 1))
        if gate_act == "softmax":
            self.gate_act = nn.Softmax(dim=1)
        else:
            self.gate_act = nn.Identity()

        self.act_name = "tanh"
        self.act = nn.Tanh()

    def _init_parameters(self, shape, dist="he") -> nn.Parameter:
        if dist == "zeros":
            return nn.Parameter(torch.zeros(*shape))

        tensor = torch.empty(*shape)
        nn.init.kaiming_normal_(tensor)
        parameter = nn.Parameter(tensor)
        return parameter

    def forward(self, x_0):
        x_l = x_0
        x_0 = x_0.unsqueeze(1)

        for layer in range(self.num_layers):
            C = self.C[layer]
            V = self.V[layer]
            U = self.U[layer]
            b_l = self.biases[layer]

            # Calculate E_i (equation 4)
            E = forward_mixture(x_l, C, U, V, self.act)
            E = E + b_l
            E = x_0 * E  # shape: batch x num_experts x hidden

            # Calculate G_i
            # num_experts x batch x 1
            gates = x_l @ self.gates
            # batch x num_experts
            gates = gates.squeeze(2).permute(1, 0)
            gates = self.gate_act(gates)

            # Calculate next x_l
            x_l = einsum(gates, E, "b e, b e d -> b d") + x_l

        return x_l


class DCNHead(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x_0):
        """
        Args:
            x_0: torch.FloatTensor - Batch x HiddenSize

        Returns:
            x_l: torch.FloatTensor - Batch x HiddenSize
        """
        x_l = x_0
        for layer in self.layers:
            x_l = x_l + x_0 * layer(x_l)
        return x_l
