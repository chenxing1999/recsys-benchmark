from typing import Dict, List, Literal, Optional, Tuple

import torch


def get_adj(
    graph: Dict[int, List[int]],
    num_item: int,
    num_user: Optional[int] = None,
    normalize=False,
) -> torch.Tensor:
    """Get Adjacency matrix from graph item"""
    if not num_user:
        num_user = max(graph.keys())

    indices: Tuple[List[int], List[int]] = ([], [])
    num_interact = 0

    for user, items in graph.items():
        indices[0].extend([user] * len(items))
        indices[1].extend(items)
        num_interact += len(items)

    indices_tensor = torch.tensor(indices)
    adj = torch.sparse_coo_tensor(
        indices_tensor,
        torch.ones(num_interact),
        size=(num_user, num_item),
    )
    if not normalize:
        return adj

    degree_user = adj.sum(dim=1).pow(-0.5)
    degree_item = adj.sum(dim=0).pow(-0.5)
    values = torch.index_select(degree_user, 0, indices_tensor[0]) * torch.index_select(
        degree_item, 0, indices_tensor[1]
    )
    adj = torch.sparse_coo_tensor(
        indices_tensor,
        values.coalesce().values(),
        size=(num_user, num_item),
    ).coalesce()

    return adj


def calculate_sparse_graph_adj_norm(
    graph: Dict[int, List[int]],
    num_item: int,
    num_user: Optional[int] = None,
    layout: Literal["coo", "csr"] = "csr",
) -> torch.Tensor:
    """Calculate A_hat from LightGCN paper based on input parameter

    Args:
        graph: Mapping from user to its list of items
        num_item
        num_user

    """
    if not num_user:
        num_user = max(graph.keys())

    indices: Tuple[List[int], List[int]] = ([], [])
    num_interact = 0
    for user, items in graph.items():
        # R
        indices[0].extend([user] * len(items))
        indices[1].extend([(item + num_user) for item in items])

        # R.T
        indices[1].extend([user] * len(items))
        indices[0].extend([(item + num_user) for item in items])

        num_interact += len(items)

    indices_tensor = torch.tensor(indices)
    adj = torch.sparse_coo_tensor(
        indices_tensor,
        torch.ones(num_interact * 2),
        size=(num_user + num_item, num_item + num_user),
    )

    degree = adj.sum(dim=0).pow(-0.5)

    values = torch.index_select(degree, 0, indices_tensor[0]) * torch.index_select(
        degree, 0, indices_tensor[1]
    )

    norm_adj = torch.sparse_coo_tensor(
        indices_tensor,
        values.coalesce().values(),
        size=(num_user + num_item, num_item + num_user),
    )
    if layout == "csr":
        norm_adj = norm_adj.to_sparse_csr()

    return norm_adj
