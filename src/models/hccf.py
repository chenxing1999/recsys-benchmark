import torch
from torch import nn

from .base import IGraphBaseCore


class HCCFModelCore(IGraphBaseCore):
    """HCCF backbone based on ..."""

    def __init__(
        self,
        num_user,
        num_item,
        num_layers=2,
        hidden_size=64,
        slope=0.5,
    ):
        super().__init__()
        self.user_emb_table = nn.Embedding(num_user, hidden_size)
        self.item_emb_table = nn.Embedding(num_item, hidden_size)
        self.activation = nn.LeakyReLU(slope)

        self.num_layers = num_layers
        self.sparse_dropout = nn.Identity()

    def get_emb_table(self, matrix):
        """
        Args:
            matrix: torch.SparseTensor (num_user, num_item)
                Normalized matrix

        Returns:
            user_emb
            item_emb
        """

        user_emb_step = self.user_emb_table.weight
        item_emb_step = self.item_emb_table.weight

        user_emb_res = user_emb_step
        item_emb_res = item_emb_step

        for layer in range(self.num_layers):
            # Style 1 - perform sparse drop single time
            sparse_drop = self.sparse_dropout(matrix)
            z_user = self.activation(sparse_drop @ item_emb_step)
            z_item = self.activation(sparse_drop.T @ user_emb_step)

            user_emb_step = z_user + user_emb_step
            user_emb_res = user_emb_res + user_emb_step

            item_emb_step = z_item + item_emb_step
            item_emb_res = item_emb_res + item_emb_step

        # Normalize for single learning rate with various layer config
        item_emb_res = item_emb_res / (self.num_layers + 1)
        user_emb_res = user_emb_res / (self.num_layers + 1)
        return user_emb_res, item_emb_res

    def get_reg_loss(self, users, pos_items, neg_items) -> torch.Tensor:
        user_emb = self.user_emb_table(users)
        pos_item_emb = self.item_emb_table(pos_items)
        neg_item_emb = self.item_emb_table(neg_items)

        reg_loss = (
            user_emb.norm(2).pow(2)
            + pos_item_emb.norm(2).pow(2)
            + neg_item_emb.norm(2).pow(2)
        ) / (2 * len(users))
        return reg_loss
