import math
from typing import List


def get_ndcg(y_pred: List[List[int]], y_true: List[List[int]], k=20) -> float:
    """Pure Python implementation of Normalized Discounted Cumulative Gain

    Args:
        y_pred: y_pred_i - Result for user i
                y_pred_ij - Item index recommended for user i with rank j

        y_true: y_true_i - Result for user i
                y_true_ij - Item index recommended for user i with rank j
    """

    ndcg = 0
    for pred_user, true_user in zip(y_pred, y_true):
        dcg = 0
        idcg = 0
        for idx, pred_item in enumerate(pred_user[:k]):
            dcg += int(pred_item in true_user) / math.log2(idx + 2)

        for idx in range(min(len(true_user), k)):
            idcg += 1 / math.log2(idx + 2)

        ndcg += dcg / idcg
    num_users = len(y_pred)
    return ndcg / num_users
