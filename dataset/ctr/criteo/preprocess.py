import argparse
import os

import torch
import tqdm
from loguru import logger
from torch.utils.data import random_split

from src.dataset.criteo.utils import get_cache_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--inp_path", "-i", default="train.txt")
    parser.add_argument("--out_folder", "-o", default=".criteo/")
    parser.add_argument("--min_threshold", "-m", default=10, type=int)
    parser.add_argument("--iter-mode", action="store_true")

    args = parser.parse_args(argv)

    assert os.path.exists(args.inp_path), f"Input path = {args.inp_path} not found"

    os.makedirs(args.out_folder, exist_ok=True)

    setattr(args, "save_line", not args.iter_mode)
    return args


args = parse_args()
# Only use train file as only it have label


inp_path = args.inp_path
fin = open(inp_path)
line_indices = []
labels = []


for idx, line in enumerate(tqdm.tqdm(fin)):
    label = line[0]

    if len(line.rstrip("\n").split("\t")) != 40:
        continue

    line_indices.append(idx)
    labels.append(label)


# Train test split with sklearn
# x_train, x_test, y_train, _ = train_test_split(
#     line_indices, labels, test_size=0.2, random_state=1, stratify=labels
# )
# x_train, x_val = train_test_split(
#     x_train, test_size=0.1, random_state=1, stratify=y_train
# )
# del y_train

# Train test split with pytorch
num_train = int(0.8 * len(line_indices))
num_val = int(0.1 * len(line_indices))
num_test = len(line_indices) - num_train - num_val
x_train, x_val, x_test = random_split(line_indices, (num_train, num_val, num_test))

x_train = list(x_train)
x_val = list(x_val)
x_test = list(x_test)

logger.info(f"n_train: {len(x_train)}, n_test: {len(x_test)}, n_val {len(x_val)}")
x_train = set(x_train)
x_test = set(x_test)
x_val = set(x_val)


logger.info("Saving list of indices for train, test, val")
tmp_path = os.path.join(args.out_folder, "train_test_val_info.bin")
torch.save(
    {
        "train": x_train,
        "test": x_test,
        "val": x_val,
    },
    tmp_path,
)

data = torch.load(tmp_path)
x_train = data["train"]
x_test = data["test"]
x_val = data["val"]


info = None
info = get_cache_data(args.inp_path, args.min_threshold)

# V1
logger.info("Creating cache file for train val test")
train_info = get_cache_data(
    args.inp_path,
    args.min_threshold,
    args.save_line,
    x_train,
)

if info:
    train_info["feat_mappers"] = info["feat_mappers"]
    train_info["defaults"] = info["defaults"]

tmp_path = os.path.join(args.out_folder, "train.bin")


torch.save(train_info, tmp_path)
del train_info

test_info = get_cache_data(
    args.inp_path,
    args.min_threshold,
    args.save_line,
    x_test,
)
test_info.pop("feat_mappers")
test_info.pop("defaults")
tmp_path = os.path.join(args.out_folder, "test.bin")
torch.save(test_info, tmp_path)
del test_info


val_info = get_cache_data(
    args.inp_path,
    args.min_threshold,
    args.save_line,
    x_val,
)
val_info.pop("feat_mappers")
val_info.pop("defaults")
tmp_path = os.path.join(args.out_folder, "val.bin")
torch.save(val_info, tmp_path)
del val_info
