"""Code used to benchmark maximum RAM usage for LightGCN inference"""
import argparse
import datetime
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import yaml  # type: ignore
from mypy_extensions import i64

from src.models import IGraphBaseCore, get_graph_model  # type: ignore


class Timer:
    forward: datetime.timedelta
    matching: datetime.timedelta
    filter_time: datetime.timedelta
    topk: datetime.timedelta

    def __init__(self):
        self.forward = datetime.timedelta()
        self.matching = datetime.timedelta()
        self.filter_time = datetime.timedelta()
        self.topk = datetime.timedelta()

    def __repr__(self):
        return (
            f"forward={self.forward}ms\n"
            f"- matching={self.matching}ms\n"
            f"- filter_time={self.filter_time}ms\n"
            f"- topk={self.topk}ms"
        )

    def merge(self, other: "Timer") -> "Timer":
        self.forward += other.forward
        self.matching += other.matching
        self.filter_time += other.filter_time
        self.topk += other.topk
        return self

    def avg(self, n_runs: int) -> "Timer":
        self.forward /= n_runs
        self.matching /= n_runs
        self.filter_time /= n_runs
        self.topk /= n_runs
        return self


@torch.no_grad()
def infer(
    model: IGraphBaseCore,
    norm_adj: torch.Tensor,
    user_id: List[i64],
    graph: Dict[i64, List[i64]],
    k: i64,
) -> Tuple[torch.return_types.topk, Timer]:
    is_cuda: bool = norm_adj.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()

    timer = Timer()
    start = datetime.datetime.now()
    user_emb, item_emb = model(norm_adj)

    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.forward = end - start
    start = end

    user_emb = user_emb[user_id]

    # Filter out item already interacted with user
    ind0: List[i64] = []
    ind1: List[i64] = []
    for idx, user in enumerate(user_id):
        ind0.extend([idx] * len(graph[user]))
        ind1.extend(graph[user])

    scores = user_emb @ item_emb.T
    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.matching = end - start
    start = end

    scores[ind0, ind1] = float("-inf")

    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.filter_time = end - start
    start = end

    y_pred = torch.topk(scores, k, dim=1)
    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.topk = end - start
    start = end

    return y_pred, timer


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "--mode",
        type=int,
        help="1: Only load data, 2: Only load model, 3: only load data + model",
        default=0,
    )
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config, args


def calculate_sparse_graph_adj_norm(
    path: str, layout: Literal["coo", "csr"] = "csr"
) -> Tuple[torch.Tensor, Dict[i64, List[i64]], i64, i64]:
    """Calculate A_hat from LightGCN paper based on input parameter

    Args:
        graph: Mapping from user to its list of items
        num_item
        num_user

    """
    graph: Dict[i64, List[i64]] = {}
    num_item: i64 = 0
    with open(path) as fin:
        for line in fin.readlines():
            info = line.strip().split()
            user_id: i64 = int(info[0])
            interacted_items: List[i64] = [int(item) for item in info[1:]]

            graph[user_id] = interacted_items
            num_item = max(*interacted_items, num_item)

    num_user = len(graph)

    # num_item in file count from 0
    num_item += 1

    indices: Tuple[List[i64], List[i64]] = ([], [])
    num_interact = 0
    for user, items in graph.items():
        # R
        indices[0].extend([user] * len(items))
        indices[1].extend([(item + num_user) for item in items])

        # R.T
        indices[1].extend([user] * len(items))
        indices[0].extend([(item + num_user) for item in items])

        num_interact += len(items)

    # cpu mem peak here
    indices_tensor = torch.tensor(indices)
    adj = torch.sparse_coo_tensor(
        indices_tensor,
        torch.ones(num_interact * 2),
        size=(num_user + num_item, num_item + num_user),
    )

    degree = adj.sum(dim=0).pow(-0.5).to_dense()
    values = torch.index_select(degree, 0, indices_tensor[0])
    values = values * torch.index_select(degree, 0, indices_tensor[1])

    norm_adj = torch.sparse_coo_tensor(
        indices_tensor,
        # values.coalesce().values(),
        values,
        size=(num_user + num_item, num_item + num_user),
    )
    if layout == "csr":
        norm_adj = norm_adj.to_sparse_csr()

    return norm_adj, graph, num_user, num_item


def main(argv: Optional[Sequence[str]] = None):
    config, args = get_config(argv)
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset_path: str = train_dataset_config["path"]

    if args.mode != 2:
        (
            norm_adj,
            graph,
            num_users,
            num_items,
        ) = calculate_sparse_graph_adj_norm(train_dataset_path)

    if args.mode == 1:
        return

    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if args.mode in [2, 3]:
        return

    # TODO: Add code for warmup / prepare if neccessary
    model.to(device)
    norm_adj = norm_adj.to(device)

    user_id: List[i64] = list(range(1))
    results, timer = infer(model, norm_adj, user_id, graph, 5)
    # End warm up

    # Start inference

    topk: i64 = 20
    n_runs: i64 = 20

    total_timer = Timer()
    for _ in range(n_runs):
        results, timer = infer(model, norm_adj, user_id, graph, topk)
        total_timer.merge(timer)
    print(total_timer.avg(n_runs))


if __name__ == "__main__":
    import os

    print(os.getpid())
    main()
    # tracemalloc.stop()
