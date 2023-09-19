from typing import List, Optional, Union

import torch

from .base import IEmbedding


class PrunedEmbedding(IEmbedding):
    """Special class wrapper to convert pruned embedding weight into the same logic
    Could only be used for inference.

    """

    def __init__(
        self,
        field_dims: Union[int, List[int]],
        hidden_size: int,
        mode: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        self._hidden_size = hidden_size
        num_item = sum(field_dims)
        weight = torch.empty((hidden_size, num_item))
        self.register_buffer("_weight", weight)

    @classmethod
    @torch.no_grad()
    def from_other_emb(cls, emb: IEmbedding, mode=None) -> "PrunedEmbedding":
        weight = emb.get_weight()
        num_item, hidden_size = weight.shape

        result = cls(num_item, hidden_size, mode)

        # weight.T --> crow_indices = num_item, col_inidices the same
        # --> weight.T make O(1) compare O(num_item)
        result._weight = weight.T.to_sparse_csr()
        return result

    def get_weight(self):
        return self._weight.T

    def forward(self, x):
        original_shape = x.shape
        result = torch.index_select(self._weight.to_sparse_coo(), 0, x.flatten())
        result = result.to_dense()
        return result.reshape(*original_shape, self._hidden_size)


if __name__ == "__main__":
    from src.models.lightgcn import LightGCN
    from src.utils import prune

    num_item, num_user = 38048, 31668
    checkpoint_path = (
        "/home/xing/workspace/phd/recsys-benchmark/checkpoints/best-sgl-wa.pth"
    )
    checkpoint = torch.load(checkpoint_path)

    model_config = checkpoint["model_config"]
    model_config.pop("name")
    model = LightGCN(num_user, num_item, **model_config)

    new_state = prune(checkpoint["state_dict"], 0.8)
    state = {
        "user_emb_table._emb_module.weight": new_state["user_emb_table.weight"],
        "item_emb_table._emb_module.weight": new_state["item_emb_table.weight"],
    }

    model.load_state_dict(state)

    model.item_emb_table = PrunedEmbedding.from_other_emb(model.item_emb_table)
    model.user_emb_table = PrunedEmbedding.from_other_emb(model.user_emb_table)

    torch.save(model.state_dict(), "checkpoints/tmp.pth")
