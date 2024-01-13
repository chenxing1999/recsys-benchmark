"""Code used to benchmark maximum RAM usage for LightGCN inference"""
import argparse
import os
import time
from typing import Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, cast

import torch
import yaml  # type: ignore
from mypy_extensions import i64

from src.metrics import get_env_metrics
from src.models import IGraphBaseCore, get_graph_model  # type: ignore
from src.models.embeddings.pruned_embedding import PrunedEmbedding
from src.models.lightgcn import LightGCN

CACHE_DATA_PATH = ".cache/data_for_infer_yelp2018.dat"


class Timer:
    forward: float
    matching: float
    filter_time: float
    topk: float

    def __init__(self):
        self.forward = 0.0
        self.matching = 0.0
        self.filter_time = 0.0
        self.topk = 0.0

    def __repr__(self):
        return (
            f"forward={self.forward * 1e3:.3f} ms\n"
            f"- matching={self.matching * 1e3:.3f} ms\n"
            f"- filter_time={self.filter_time * 1e3:.3f} ms\n"
            f"- topk={self.topk * 1e3:.3f} ms"
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


# @torch.no_grad()
# @profile
def infer(
    model: IGraphBaseCore,
    norm_adj: torch.Tensor,
    user_id: List[i64],
    graph: Dict[i64, List[i64]],
    k: i64,
) -> Tuple[NamedTuple, Timer]:
    is_cuda: bool = norm_adj.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()

    timer = Timer()
    start = time.time()
    with torch.no_grad():
        user_emb, item_emb = model(norm_adj)

    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    timer.forward = end - start
    start = end

    user_emb = user_emb[user_id]

    scores = user_emb @ item_emb.T
    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    timer.matching = end - start
    start = end

    # Filter out item already interacted with user
    # ind0: List[i64] = []
    # ind1: List[i64] = []
    # for idx, user in enumerate(user_id):
    #     ind0.extend([idx] * len(graph[user]))
    #     ind1.extend(graph[user])
    ind0 = torch.tensor([], dtype=torch.long)
    ind1 = torch.tensor([], dtype=torch.long)
    for idx, user in enumerate(user_id):
        ind0 = torch.cat((ind0, torch.tensor([idx] * len(graph[user]))))
        ind1 = torch.cat((ind1, torch.tensor(graph[user])))

    scores[ind0, ind1] = float("-inf")

    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    timer.filter_time = end - start
    start = end

    y_pred = torch.topk(scores, k, dim=1)
    if is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    timer.topk = end - start
    start = end

    return y_pred, timer


def get_config(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "--task",
        "-t",
        type=int,
        help="""
        1: Only load data,
        2: load data + model,
        3: do nothing,
        4: Load model then save to binary
        0 (default): infer
        """,
        default=0,
    )
    parser.add_argument(
        "--save-binary-path",
        "-s",
        type=str,
        help="path to binary file to save in task 4",
        default=None,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Model loader, support: original, pep",
        default="original",
    )
    parser.add_argument(
        "--n_runs",
        "-n",
        type=int,
        help="Number of runs. Defaults: 20",
        default=20,
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


def _load_pep(
    config,
    num_users,
    num_items,
    device: str = "cpu",
):
    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )
    model = cast(LightGCN, model)

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.item_emb_table = PrunedEmbedding.from_other_emb(model.item_emb_table)
    model.user_emb_table = PrunedEmbedding.from_other_emb(model.user_emb_table)

    if device == "cuda":
        model.item_emb_table.to_cuda()
        model.user_emb_table.to_cuda()
    return model


def _load_ttrec(
    config,
    num_users,
    num_items,
    device: str = "cpu",
    cache=False,
):
    """Utility to load TTRec checkpoint with cache or not"""
    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    config["model"]["embedding_config"]["use_cache"] = cache

    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    if not cache:
        for k in ["item_emb_table", "user_emb_table"]:
            checkpoint["state_dict"].pop(f"{k}._tt_emb.hashtbl")
            checkpoint["state_dict"].pop(f"{k}._tt_emb.cache_state")

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # if cache:
    #     model.embedding._tt_emb.cache_populate()
    return model


def _load_optembed(
    config,
    num_users,
    num_items,
    device: str = "cpu",
):
    from src.models.embeddings.lightgcn_opt_embed import RetrainOptEmbed

    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    embs = [model.item_emb_table, model.user_emb_table]
    embs = cast(List[RetrainOptEmbed], embs)

    for emb in embs:
        emb._cur_weight = emb._weight * emb._mask
        emb._cur_weight = emb._cur_weight.to(device)
        emb._full_mask_d = None  # type: ignore
        emb._mask = None  # type: ignore
        emb._weight = None  # type: ignore

    model.to(device)

    return model


def _load_optembed_sparse(
    config,
    num_users,
    num_items,
    device: str = "cpu",
):
    pass

    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.item_emb_table = PrunedEmbedding.from_other_emb(model.item_emb_table)
    model.user_emb_table = PrunedEmbedding.from_other_emb(model.user_emb_table)

    if device == "cuda":
        model.item_emb_table.to_cuda()
        model.user_emb_table.to_cuda()

    return model


def _load_original(config, num_users, num_items, device):
    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # TODO: Add code for warmup / prepare if neccessary
    model.to(device)
    return model


def _load_cerp(config, num_users, num_items, device):
    model_config = config["model"]
    model = get_graph_model(
        num_users,
        num_items,
        model_config,
    )

    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    for emb in [model.item_emb_table, model.user_emb_table]:
        emb.sparse_p_weight = emb.p_weight * emb.p_mask
        emb.sparse_q_weight = emb.q_weight * emb.q_mask

        emb.p_weight = None
        emb.q_weight = None
        emb.q_mask = None
        emb.p_mask = None

    model.to(device)

    return model


def main(argv: Optional[Sequence[str]] = None):
    config, args = get_config(argv)
    if args.task == 3:
        return
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset_path: str = train_dataset_config["path"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if os.path.exists(CACHE_DATA_PATH):
        (
            norm_adj,
            graph,
            num_users,
            num_items,
        ) = torch.load(CACHE_DATA_PATH)
    else:
        (
            norm_adj,
            graph,
            num_users,
            num_items,
        ) = calculate_sparse_graph_adj_norm(train_dataset_path)
        dat = (
            norm_adj,
            graph,
            num_users,
            num_items,
        )
        os.makedirs(os.path.dirname(CACHE_DATA_PATH), exist_ok=True)
        torch.save(dat, CACHE_DATA_PATH)

    norm_adj = norm_adj.to(device)
    if args.task == 1:
        print(get_env_metrics())
        return

    if args.mode in ["original", "qr", "dhe", "tt_rec_torch"]:
        model = _load_original(config, num_users, num_items, device)
    elif args.mode == "pep":
        model = _load_pep(config, num_users, num_items, device)
    elif args.mode == "ttrec":
        model = _load_ttrec(config, num_users, num_items, device)
    elif args.mode == "optemb":
        model = _load_optembed(config, num_users, num_items, device)
    elif args.mode == "optemb_sparse":
        model = _load_optembed_sparse(config, num_users, num_items, device)
    elif args.mode == "cerp":
        model = _load_cerp(config, num_users, num_items, device)
    elif args.mode == "torch":
        model = torch.load(config["checkpoint_path"])
    else:
        raise ValueError(f"{args.mode=} not supported")
    if args.task == 2:
        print(get_env_metrics())
        return

    if args.task == 4:
        torch.save(model, args.save_binary_path)
        return

    model.eval()
    with torch.no_grad():
        model(norm_adj)[0].cpu()

    user_id: List[i64] = list(range(1))
    results, timer = infer(model, norm_adj, user_id, graph, 5)
    # End warm up

    # Start inference

    topk: i64 = 20
    n_runs: i64 = args.n_runs

    total_timer = Timer()
    for _ in range(n_runs):
        results, timer = infer(model, norm_adj, user_id, graph, topk)
        total_timer.merge(timer)

    print("Timer run")
    print(total_timer.avg(n_runs))

    print("Env Metric")
    print(get_env_metrics())


if __name__ == "__main__":
    main()
