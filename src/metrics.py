import os
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import psutil
import torch


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

    ndcg: float = 0
    for pred_user, true_user in zip(y_pred, y_true):
        dcg: np.ndarray
        idcg: np.ndarray

        dcg_sum: float
        idcg_sum: float

        dcg = np.array(
            list(map(lambda pred_item: pred_item in true_user, pred_user[:k]))
        )
        weight = 1 / np.log2(np.arange(2, dcg.shape[0] + 2))
        dcg_sum = (weight * dcg).sum()

        length = min(len(true_user), k)
        idcg = 1 / (np.log2(np.arange(2, length + 2)))
        idcg_sum = idcg.sum()

        ndcg += dcg_sum / idcg_sum
    num_users = len(y_pred)
    return ndcg / num_users


def get_env_metrics() -> Dict[str, float]:
    """Utils function for quick benchmark performance

    Returns: a metric dictionary that contains
        various computation wise metrics such as cpu mem and peak CUDA mem

    Note: Copied and modified from RecBole
    """

    memory_used = psutil.Process(os.getpid()).memory_info().rss
    cpu_usage = psutil.cpu_percent(interval=1)
    peak_cuda_mem = torch.cuda.max_memory_allocated()
    cur_cuda_mem = torch.cuda.memory_allocated()

    return {
        "cur_cpu_memory": memory_used,
        "cur_cpu_usage": cpu_usage,
        "cur_cuda_mem": cur_cuda_mem,
        "peak_cuda_mem": peak_cuda_mem,
    }


def get_ndcg_recall(
    y_pred: List[List[int]],
    y_true: List[Union[List[int], Set[int]]],
    k=20,
) -> Tuple[float, float]:
    """Get ndcg and recall

    Args:
        y_pred: y_pred_i - Result for user i
                y_pred_ij - Item index recommended for user i with rank j

        y_true: y_true_i - Result for user i
                y_true_ij - if list, Item index recommended for user i with rank j
    """

    ndcg: float = 0
    recall: float = 0
    for pred_user, true_user in zip(y_pred, y_true):
        dcg: float
        idcg: float

        # array with shape K
        is_relavent = np.array([pred_item in true_user for pred_item in pred_user[:k]])

        weight = 1 / np.log2(np.arange(2, is_relavent.shape[0] + 2))
        dcg = (weight * is_relavent).sum()

        length = min(len(true_user), k)
        idcg = (1 / (np.log2(np.arange(2, length + 2)))).sum()

        ndcg += dcg / idcg

        num_correct = is_relavent.sum()
        recall += num_correct / length

    ndcg /= len(y_pred)
    recall /= len(y_pred)
    return ndcg, recall
