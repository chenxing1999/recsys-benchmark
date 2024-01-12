import argparse
import time
from pprint import pprint
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.dataset.criteo.utils import NUM_FEATS, preprocess
from src.metrics import get_env_metrics
from src.models.deepfm import DeepFM
from src.models.embeddings.pruned_embedding import PrunedEmbedding
from src.utils import set_seed

set_seed(2023)

TEST_CODE = """
with torch.no_grad():
    model(inps).cpu()
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        "-c",
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--train-info",
        help="Path to train info, used to load feat_mappers and defaults",
    )

    parser.add_argument("--batch_size", "-b", default=64, type=int)
    parser.add_argument(
        "--task",
        "-t",
        help="Which task to run (infer, load_model, load_data, ranking"
        ", nothing, timeit)",
        default="infer",
    )
    parser.add_argument(
        "--run-small-file",
        action="store_true",
        help="Run small file (100 record) in tests/assets instead of full data",
    )

    parser.add_argument(
        "--mode",
        "-m",
        default="original",
        help="Mode to load model weight",
    )

    args = parser.parse_args(argv)
    # assert args.mode in MODES

    return args


def read_small_file(
    fpath: str,
    batch_size: int,
    feat_mappers: Dict,
    defaults: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(fpath) as fin:
        lines = fin.readlines()
        lines = [line.rstrip("\n") for line in lines]

    train_info = {
        "feat_mappers": feat_mappers,
        "defaults": defaults,
    }

    num_line = len(lines)

    res: List[Tuple[torch.Tensor, int]]
    if num_line < batch_size:
        res = [preprocess(train_info, line) for line in lines]

        res = [res[i % num_line] for i in range(batch_size)]
    else:
        res = [preprocess(train_info, lines[i]) for i in range(batch_size)]

    inps, labels = zip(*res)

    inps_tensor = torch.stack(inps, dim=0)
    label_tensor = torch.tensor(labels)
    return inps_tensor, label_tensor


def read_big_file(
    fpath: str,
    batch_size: int,
    feat_mappers: Dict,
    defaults: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inps = []
    labels = []

    train_info = {
        "feat_mappers": feat_mappers,
        "defaults": defaults,
    }
    count = 0
    with open(fpath) as fin:
        for idx, line in enumerate(fin):
            inp = fin.readline()

            inp = inp.rstrip("\n").split("\t")
            if len(inp) != NUM_FEATS + 1:
                continue

            inp_tensor, label = preprocess(train_info, inp)
            labels.append(label)
            inps.append(inp_tensor)
            count += 1
            if count == batch_size:
                break

            if count % 100 == 0:
                print(f"{count} / {batch_size}")

    inps_tensor = torch.stack(inps, dim=0)
    label_tensor = torch.tensor(labels)
    return inps_tensor, label_tensor


def _load_dhe(checkpoint_path: str):
    """Remove DHE cache mode before loading"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_config"]["embedding_config"]["cached"] = False

    return DeepFM.load(checkpoint)


def _load_pep(checkpoint_path: str):
    """convert pep to pruned version"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint["model_config"]["embedding_config"]["name"] != "pruned-sparse-csr":
        model = DeepFM.load(checkpoint, strict=False)
        model.embedding = PrunedEmbedding.from_other_emb(model.embedding)
        return model

    sparse_w = checkpoint["state_dict"].pop("embedding.sparse_w")
    model = DeepFM.load(checkpoint, empty_embedding=True)
    # setattr(model, "embedding", PrunedEmbedding.from_weight(sparse_w))
    model.register_module("embedding", PrunedEmbedding.from_weight(sparse_w))
    return model


def _load_ttrec(checkpoint_path: str, cache=False):
    """Utility to load TTRec checkpoint with cache or not"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_config"]["embedding_config"]["use_cache"] = cache

    if not cache:
        checkpoint["state_dict"].pop("embedding._tt_emb.hashtbl")
        checkpoint["state_dict"].pop("embedding._tt_emb.cache_state")

    model = DeepFM.load(checkpoint, strict=False)
    model.to("cuda")
    if cache:
        # model.embedding._tt_emb.cache_populate()
        model.embedding._tt_emb.warmup = False
    return model


def _load_opt_mask_d(
    checkpoint_path: str,
    initial_path: str,
    mem_optimized=False,
) -> DeepFM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = DeepFM(checkpoint["field_dims"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # load mask_d
    print("load mask d")
    initial_checkpoint = torch.load(initial_path, map_location="cpu")
    mask = initial_checkpoint["mask"]["mask_d"]
    weight = model.embedding.get_weight(mask)

    # Remove original weight
    model.embedding._weight.data = torch.empty(0)

    if mem_optimized:
        model.embedding = PrunedEmbedding.from_weight(weight)
    return model


def _load_cerp(checkpoint_path):
    import numpy as np

    from src.models.embeddings.cerp_embedding import RetrainCerpEmbedding

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DeepFM.load(checkpoint, empty_embedding=True, strict=False)

    emb_config = checkpoint["model_config"]["embedding_config"]

    bucket_size = emb_config["bucket_size"]
    emb_config["bucket_size"] = 1
    emb_config.pop("name")
    emb = RetrainCerpEmbedding(
        checkpoint["field_dims"],
        16,
        mode=None,
        field_name="deepfm",
        **emb_config,
    )

    emb.p_weight = None
    emb.q_weight = None
    emb.p_mask = None
    emb.q_mask = None

    state = checkpoint["state_dict"]
    emb.sparse_p_weight = state["embedding.sparse_p_weight"]
    emb.sparse_q_weight = state["embedding.sparse_q_weight"]

    emb._bucket_size = bucket_size
    emb.q_entity_per_row = int(np.ceil(emb._num_item / emb._bucket_size))
    model.embedding = emb

    # model = DeepFM.load(checkpoint_path)

    # emb = model.embedding
    # emb.sparse_p_weight = emb.p_weight * emb.p_mask
    # emb.sparse_q_weight = emb.q_weight * emb.q_mask

    # emb.p_weight = None
    # emb.q_weight = None
    # emb.q_mask = None
    # emb.p_mask = None

    return model


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    if args.task == "nothing":
        return

    #  Load data
    train_info = torch.load(args.train_info)
    if args.task == "load_data":
        return

    # Load checkpoint
    if args.mode in ["original", "opt-cpu"]:
        model = torch.load(args.checkpoint_path)
    elif args.mode in ["original", "qr", "ttrec-cpu", "opt-cpu"]:
        model = DeepFM.load(args.checkpoint_path)
    elif args.mode == "cerp":
        model = _load_cerp(args.checkpoint_path)
    elif args.mode == "dhe":
        model = _load_dhe(args.checkpoint_path)
    elif args.mode in ["pep", "opt-sparse-cpu"]:
        model = _load_pep(args.checkpoint_path)
    elif args.mode == "ttrec":
        model = _load_ttrec(args.checkpoint_path, True)
    elif args.mode == "opt":
        model = _load_opt_mask_d(
            args.checkpoint_path,
            "checkpoints/deepfm/opt/initial.pth",
            mem_optimized=False,
        )
    else:
        raise ValueError()

    if args.task == "load_model":
        return

    print("Loading...")
    print("Num params", model.embedding.get_num_params())
    if args.run_small_file:
        inps, labels = read_small_file(
            "tests/assets/train_criteo_sample.txt",
            args.batch_size,
            train_info["feat_mappers"],
            train_info["defaults"],
        )
    else:
        inps, labels = read_big_file(
            "dataset/ctr/criteo/train.txt",
            args.batch_size,
            train_info["feat_mappers"],
            train_info["defaults"],
        )

    print(inps.shape)
    # infer
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.embedding.sparse_p_weight = model.embedding.sparse_p_weight.to(device)
    model.embedding.sparse_q_weight = model.embedding.sparse_q_weight.to(device)
    if isinstance(model.embedding, PrunedEmbedding) and device == "cuda":
        model.embedding.to_cuda()

    print("--- Load model ---")
    if torch.cuda.is_available():
        # wait for data move to gpu
        torch.cuda.synchronize()
    pprint(get_env_metrics())

    inps = inps.to(device)

    # warmp up
    with torch.no_grad():
        model(inps)

    if torch.cuda.is_available():
        # wait for data move to gpu
        torch.cuda.synchronize()

    if args.task == "ranking":
        print("--- Ranking ---")
        start = time.time()
        with torch.no_grad():
            model.get_ranks(inps).cpu()
    elif args.task == "timeit":
        import timeit

        start = time.time()
        print(
            timeit.repeat(
                TEST_CODE,
                setup="import torch",
                globals={"model": model, "inps": inps},
                number=20,
            )
        )

    elif args.task == "track_emission":
        raise ValueError("Track emission is not supported")
    else:
        print("--- Inference ---")
        start = time.time()
        with torch.no_grad():
            model(inps).cpu()

    print(time.time() - start)

    # print(torch.sigmoid(out).to(torch.int32).tolist());
    # print(labels)
    # print(roc_auc_score(labels, out))

    pprint(get_env_metrics())


if __name__ == "__main__":
    main()
