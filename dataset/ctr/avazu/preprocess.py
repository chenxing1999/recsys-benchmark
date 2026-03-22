import argparse
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Set

import torch
from loguru import logger
from mypy_extensions import u8
from torch.utils.data import random_split

from src.dataset.avazu.utils import preprocess_line, run_timestamp_preprocess
from src.utils import set_seed

set_seed(2023)
NUM_FEATS = 22

# no missing data
N_DATA = 40428967


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--inp_path", "-i", default="train")
    parser.add_argument("--out_folder", "-o", default=".avazu/")
    parser.add_argument("--min_threshold", "-m", default=10, type=int)
    parser.add_argument("--iter-mode", action="store_true")
    parser.add_argument("--preprocess_timestamp", action="store_true")

    args = parser.parse_args(argv)

    assert os.path.exists(args.inp_path), f"Input path = {args.inp_path} not found"

    os.makedirs(args.out_folder, exist_ok=True)

    setattr(args, "save_line", not args.iter_mode)
    return args


def get_cache_data(
    path: str,
    min_threshold: int = 10,
    save_line=False,
    lines: Optional[Set[int]] = None,
    preprocess_timestamp: bool = False,
) -> Dict[str, Any]:
    """Get cache object

    Args:
        path: Path to original train.txt / test.txt file
            of CriteoDataset

        min_threshold

        save_line

        lines: List of line idx will be dataset

    Returns: Dict[str, Any]

        feat_mappers: (FeatMapper)
            feat_mapper[feat_idx][feat_value] = feat_id corresponding to feature_idx

        defaults (Dict[int, int])
            defaults[feat_idx] = default value for OOV feature of feat_idx

        num_data (int)
        line_idx_to_byte_loc: List[int]:
    """

    feat_cnts: DefaultDict[u8, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    counts = 0
    line_idx_to_byte_loc = []
    with open(path) as fin:
        loc = 0
        fin.readline()  # skip header
        for idx, line in enumerate(fin):
            loc += len(line)
            values = line.rstrip("\n").split(",")

            if len(values) != NUM_FEATS + 2:
                continue

            extra_feats = []
            if preprocess_timestamp:
                extra_feats = run_timestamp_preprocess(values)

            # convert timestamp to hour
            preprocess_line(values)
            values.extend(extra_feats)

            num_feats = NUM_FEATS + len(extra_feats)

            for i in range(1, num_feats + 1):
                # skip label
                feat_cnts[i][values[i + 1]] += 1

            if save_line:
                line_idx_to_byte_loc.append(loc - len(line))
            counts += 1

    # feat_idx_to_set map from FeatureIndex to Set of feature values
    feat_idx_to_set: Dict[u8, Set[str]] = {
        i: {feat for feat, c in cnt.items() if c >= min_threshold}
        for i, cnt in feat_cnts.items()
    }

    feat_mappers: Dict[u8, Dict[str, int]] = {
        i: {feat: idx for idx, feat in enumerate(cnt)}
        for i, cnt in feat_idx_to_set.items()
    }
    defaults = {i: len(cnt) for i, cnt in feat_mappers.items()}

    info = {
        "feat_mapper": feat_mappers,
        "defaults": defaults,
        "num_data": counts,
        "metadata": dict(
            preprocess_timestamp=preprocess_timestamp,
            version=2,
        ),
    }

    if save_line:
        info["line_idx_to_byte_loc"] = line_idx_to_byte_loc
    return info


args = parse_args()


inp_path = args.inp_path
fin = open(inp_path)

# import tqdm
# for idx, line in enumerate(tqdm.tqdm(fin)):
#     label = line[0]
#
#     if len(line.rstrip("\n").split("\t")) != 12:
#         continue
#
#     line_indices.append(idx)
#     labels.append(label)


# Train test split with sklearn
# x_train, x_test, y_train, _ = train_test_split(
#     line_indices, labels, test_size=0.2, random_state=1, stratify=labels
# )
# x_train, x_val = train_test_split(
#     x_train, test_size=0.1, random_state=1, stratify=y_train
# )
# del y_train
# line_indices = cast(Dataset[int], line_indices)

# Train test split with pytorch
line_indices = list(range(N_DATA))
print(len(line_indices))
num_train = int(0.8 * len(line_indices))
num_val = int(0.1 * len(line_indices))
num_test = len(line_indices) - num_train - num_val

generator = torch.Generator()
generator.manual_seed(1)
x_train, x_val, x_test = random_split(
    line_indices, (num_train, num_val, num_test), generator
)

logger.info(f"n_train: {len(x_train)}, n_test: {len(x_test)}, n_val {len(x_val)}")
x_train = set(x_train)
x_test = set(x_test)
x_val = set(x_val)


logger.info("Saving list of indices for train, test, val")
tmp_path = os.path.join(args.out_folder, "train_test_info.bin")
info = {
    "train": x_train,
    "test": x_test,
    "val": x_val,
}
torch.save(
    info,
    tmp_path,
)

# data = torch.load(tmp_path)
# x_train = data["train"]
# x_test = data["test"]
# x_val = data["val"]


print("start preprocessing")
info2 = get_cache_data(args.inp_path, args.min_threshold)
info.update(info2)

tmp_path = os.path.join(args.out_folder, "train_test_info.bin")
torch.save(info, tmp_path)

# V1
# logger.info("Creating cache file for train val test")
# train_info = get_cache_data(
#     args.inp_path,
#     args.min_threshold,
#     args.save_line,
#     x_train,
# )
#
# if info:
#     train_info["feat_mappers"] = info["feat_mappers"]
#     train_info["defaults"] = info["defaults"]
#
# tmp_path = os.path.join(args.out_folder, "train.bin")
#
#
# torch.save(train_info, tmp_path)
# del train_info
#
# test_info = get_cache_data(
#     args.inp_path,
#     args.min_threshold,
#     args.save_line,
#     x_test,
# )
# test_info.pop("feat_mappers")
# test_info.pop("defaults")
# tmp_path = os.path.join(args.out_folder, "test.bin")
# torch.save(test_info, tmp_path)
# del test_info
#
#
# val_info = get_cache_data(
#     args.inp_path,
#     args.min_threshold,
#     args.save_line,
#     x_val,
# )
# val_info.pop("feat_mappers")
# val_info.pop("defaults")
# tmp_path = os.path.join(args.out_folder, "val.bin")
# torch.save(val_info, tmp_path)
# del val_info
