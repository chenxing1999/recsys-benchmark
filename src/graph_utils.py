from typing import Dict, List, Optional

import torch


def calculate_sparse_graph_adj_norm(
    graph: Dict[int, List[int]],
    num_item: int,
    num_user: Optional[int] = None,
) -> torch.Tensor:
    """Calculate  A_hat

    Args:
        graph: Mapping from user to its list of items
        num_item
        num_user

    """
    if not num_user:
        num_user = max(graph.keys())

    indices = [[], []]
    num_interact = 0
    for user, items in graph.items():
        # R
        indices[0].extend([user] * len(items))
        indices[1].extend([(item + num_user) for item in items])

        # R.T
        indices[1].extend([user] * len(items))
        indices[0].extend([(item + num_user) for item in items])

        num_interact += len(items)

        # norm_degree = 1 / math.sqrt(len(items))
        # degrees.append(norm_degree)

    adj = torch.sparse_coo_tensor(
        indices,
        torch.ones(num_interact * 2),
        size=(num_user + num_item, num_item + num_user),
    )

    degree = adj.sum(dim=0).pow(-0.5)

    indicies = adj.coalesce().indices()
    values = torch.index_select(degree, 0, indicies[0]) * torch.index_select(
        degree, 0, indicies[1]
    )

    norm_adj = torch.sparse_coo_tensor(
        indices,
        values.coalesce().values(),
        size=(num_user + num_item, num_item + num_user),
    )

    return norm_adj
