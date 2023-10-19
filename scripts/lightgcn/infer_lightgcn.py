"""Code used to benchmark maximum RAM usage for LightGCN inference"""
import argparse
import datetime
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# tracemalloc.start()
import torch
import yaml

from src.dataset.cf_graph_dataset import CFGraphDataset
from src.models import get_graph_model


# Timer = namedtuple("Timer", ["forward", "filter", "topk"])
@dataclass
class Timer:
    forward = 0
    matching = 0
    filter_time = 0
    topk = 0

    def __repr__(self):
        return (
            f"forward={self.forward}ms\n"
            f"- matching={self.matching}ms\n"
            f"- filter_time={self.filter_time}ms\n"
            f"- topk={self.topk}ms"
        )

    def merge(self, other):
        self.forward += other.forward
        self.matching += other.matching
        self.filter_time += other.filter_time
        self.topk += self.topk
        return self

    def avg(self, n_runs):
        self.forward /= n_runs
        self.matching /= n_runs
        self.filter_time /= n_runs
        self.topk /= n_runs
        return self


@torch.no_grad()
def infer(model, norm_adj, user_id, graph, k):
    if isinstance(user_id, int):
        user_id = [user_id]
    user_id: List[int]

    is_cuda = norm_adj.device.type == "cuda"
    print("is cuda", is_cuda)
    if is_cuda:
        torch.cuda.synchronize()

    timer = Timer()
    start = datetime.datetime.now()
    user_emb, item_emb = model(norm_adj)

    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.forward = (end - start).microseconds
    start = end

    user_emb = user_emb[user_id]

    # Filter out item already interacted with user
    ind0 = []
    ind1 = []
    for idx, user in enumerate(user_id):
        ind0.extend([idx] * len(graph[user]))
        ind1.extend(graph[user])

    scores = user_emb @ item_emb.T
    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.matching = (end - start).microseconds
    start = end

    scores[ind0, ind1] = float("-inf")

    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.filter_time = (end - start).microseconds
    start = end

    y_pred = torch.topk(scores, k, dim=1)
    if is_cuda:
        torch.cuda.synchronize()
    end = datetime.datetime.now()
    timer.topk = (end - start).microseconds
    start = end

    return y_pred, timer


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def main(argv=None):
    config = get_config(argv)
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
    print("Successfully load train dataset")
    train_dataset.describe()

    norm_adj = train_dataset.get_norm_adj()
    graph = train_dataset.get_graph()

    model_config = config["model"]
    model = get_graph_model(
        train_dataset.num_users,
        train_dataset.num_items,
        model_config,
    )
    del train_dataset

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # TODO: Add code for warmup / prepare if neccessary
    model.to(device)
    norm_adj = norm_adj.to(device)
    print(sys.getsizeof(norm_adj))

    user_id = 1
    results, timer = infer(model, norm_adj, user_id, graph, 5)
    # End warm up

    # Start inference

    topk = 20
    # user_id = list(range(1024))
    user_id = list(range(1))
    n_runs = 20

    total_timer = Timer()
    for _ in range(n_runs):
        results, timer = infer(model, norm_adj, user_id, graph, topk)
        total_timer.merge(timer)
    # print(tracemalloc.get_traced_memory())
    # print(results)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    print(total_timer.avg(n_runs))


if __name__ == "__main__":
    import os

    print(os.getpid())
    input()
    main()
    # tracemalloc.stop()
    input()
