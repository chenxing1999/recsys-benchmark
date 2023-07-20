from typing import List, Set, Union

import numpy as np


def get_ndcg(
    y_pred: List[List[int]],
    y_true: List[Union[List[int], Set[int]]],
    k=20,
) -> float:
    """Python implementation of Normalized Discounted Cumulative Gain

    Args:
        y_pred: y_pred_i - Result for user i
                y_pred_ij - Item index recommended for user i with rank j

        y_true: y_true_i - Result for user i
                y_true_ij - if list, Item index recommended for user i with rank j
    """

    ndcg = 0
    for pred_user, true_user in zip(y_pred, y_true):
        dcg = 0
        idcg = 0

        dcg = np.array(
            list(map(lambda pred_item: pred_item in true_user, pred_user[:k]))
        )
        weight = 1 / np.log2(np.arange(2, dcg.shape[0] + 2))
        dcg = (weight * dcg).sum()

        length = min(len(true_user), k)
        idcg = 1 / (np.log2(np.arange(2, length + 2)))
        idcg = idcg.sum()

        ndcg += dcg / idcg
    num_users = len(y_pred)
    return ndcg / num_users
