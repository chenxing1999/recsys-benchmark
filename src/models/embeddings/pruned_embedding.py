from typing import List, Optional, Union

import numba
import numpy as np
import torch
from numba import cuda

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
        self.is_cuda = False
        self._num_item = num_item

    @classmethod
    @torch.no_grad()
    def from_other_emb(cls, emb: IEmbedding, mode=None) -> "PrunedEmbedding":
        return cls.from_weight(emb.get_weight(), mode)

    @classmethod
    def from_weight(cls, weight: torch.Tensor, mode=None) -> "PrunedEmbedding":
        num_item, hidden_size = weight.shape
        result = cls(num_item, hidden_size, mode)

        weight = weight.to_sparse_csr()
        result.values = weight.values().numpy()

        crow_indices = weight.crow_indices()
        result.crow_indices = crow_indices.numpy()

        col_indices = weight.col_indices()
        result.col_indices = col_indices.numpy()

        return result

    def to_cuda(self):
        """Convert Pruned Embedding from CPU to CUDA"""
        if self.is_cuda:
            return

        # self.values = numba.cuda.to_device(self.values)
        # self.crow_indices = numba.cuda.to_device(self.crow_indices)
        # self.col_indices = numba.cuda.to_device(self.col_indices)

        # simply convert numpy array to numba will make PyTorch API cannot
        # keep track GPU memory usage of Numba
        self.values = self._to_cuda(self.values)
        self.crow_indices = self._to_cuda(self.crow_indices)
        self.col_indices = self._to_cuda(self.col_indices, sync=True)
        self.is_cuda = True

    @staticmethod
    def _to_cuda(np_arr: np.ndarray, sync=False):
        return numba.cuda.as_cuda_array(torch.from_numpy(np_arr).to("cuda"), sync=sync)

    def get_weight(self):
        sparse_csr = torch.sparse_csr_tensor(
            self._to_torch(self.crow_indices),
            self._to_torch(self.col_indices),
            self._to_torch(self.values),
            size=(self._num_item, self._hidden_size),
            dtype=torch.float32,
        )
        return sparse_csr.to_dense()

    def _to_torch(self, x) -> torch.Tensor:
        """Convert x into torch.Tensor"""
        if self.is_cuda:
            return torch.as_tensor(x, device="cuda")
        else:
            return torch.from_numpy(x)

    def forward(self, x):
        original_shape = x.shape

        if self.is_cuda:
            x = x.flatten()
            tensor_size = x.shape[0]
            x = numba.cuda.as_cuda_array(x)
            hidden_size = self._hidden_size

            n_threads = 32
            n_blocks = (tensor_size - 1) // 32 + 1
            results = torch.zeros(
                (tensor_size, hidden_size), dtype=torch.float32, device="cuda"
            )
            results = numba.cuda.as_cuda_array(results)

            csr_embedding_lookup[n_blocks, n_threads](
                self.values,
                self.crow_indices,
                self.col_indices,
                x,
                results,
                tensor_size,
                hidden_size,
            )
            results = torch.as_tensor(results, device="cuda")
            return results.reshape(*original_shape, self._hidden_size)
        else:
            x = x.flatten().numpy()
            tensor_size = x.shape[0]
            hidden_size = self._hidden_size

            results = np.empty((tensor_size, hidden_size), dtype=np.float32)

            csr_embedding_lookup_cpu(
                self.values,
                self.crow_indices,
                self.col_indices,
                x,
                results,
                tensor_size,
                hidden_size,
            )
            results = torch.as_tensor(results)
            return results.reshape(*original_shape, self._hidden_size)


@cuda.jit
def csr_embedding_lookup(
    values,
    crow_indices,
    col_indices,
    ids,
    outputs,
    size,
    hidden_size,
):
    """
    Get value from

    Args:
        values
        crow_indices
        col_indices
        ids: Index to select from
        outputs: to store output
            shape: N x D

        size: ids size
        hidden_size: Size of final dimension of outputs
    """
    index = cuda.grid(1)

    if index >= size:
        return

    rowid = ids[index]
    left = crow_indices[rowid]
    right = crow_indices[rowid + 1]

    for i in range(hidden_size):
        outputs[index][i] = 0

    for i in range(left, right):
        outputs[index][col_indices[i]] = values[i]


@numba.jit(nopython=True)
def csr_embedding_lookup_cpu(
    values,
    crow_indices,
    col_indices,
    ids,
    outputs,
    size,
    hidden_size,
):
    left = crow_indices[ids]
    right = crow_indices[ids + 1]

    outputs[:] = 0

    for index in range(size):
        for i in range(left[index], right[index]):
            outputs[index][col_indices[i]] = values[i]


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
