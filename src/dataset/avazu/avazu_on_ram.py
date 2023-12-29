"""Naively load all data to RAM because you can for speeding up training"""

from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Final, List, Literal, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from src.dataset.avazu.utils import run_timestamp_preprocess
from src.dataset.base import ICTRDataset
from src.dataset.criteo.utils import merge_feat_mapper_default

NUM_FEATS: Final[int] = 22


def _create_binary(
    txt_path: str,
    min_threshold: int = 2,
    seed=2023,
    split_strategy=1,
    preprocess_timestamp: bool = False,
) -> Dict[str, Any]:
    """
    Args:

        split_strategy:
            0: Split by time based
            1: random split
    Note: Copy from pytorch-fm with modification
    """
    NUM_FEATS = 22
    metadata = {
        "txt_path": txt_path,
        "seed": seed,
        "min_threshold": min_threshold,
        "split_strategy": split_strategy,
        "preprocess_timestamp": preprocess_timestamp,
    }

    feat_cnts: DefaultDict[DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    line_indices = []
    labels = []
    with open(txt_path) as f:
        f.readline()  # skip header

        pbar = tqdm(enumerate(f), mininterval=1, smoothing=0.1)
        pbar.set_description("Create avazu dataset cache: counting features")
        for idx, line in pbar:
            values = line.rstrip("\n").split(",")
            if len(values) != NUM_FEATS + 2:
                continue
            for i in range(1, NUM_FEATS + 1):
                feat_cnts[i][values[i + 1]] += 1

            label = int(values[1])
            line_indices.append(idx)
            labels.append(label)
            if preprocess_timestamp:
                extra_feats = run_timestamp_preprocess(values)
                for i, feat in enumerate(extra_feats):
                    feat_cnts[NUM_FEATS + 1 + i][feat] += 1

    # Note: Use list instead of set compare to original pytorch-fm
    # because list is deterministic
    idx_to_list_feat: Dict[int, List[str]] = {
        i: [feat for feat, c in cnt.items() if c >= min_threshold]
        for i, cnt in feat_cnts.items()
    }

    feat_mapper: Dict[int, Dict[str, int]] = {
        i: {feat: idx for idx, feat in enumerate(cnt)}
        for i, cnt in idx_to_list_feat.items()
    }
    defaults: Dict[int, int] = {i: len(cnt) for i, cnt in feat_mapper.items()}

    num_train = int(0.8 * len(line_indices))
    num_val = int(0.1 * len(line_indices))
    num_test = len(line_indices) - num_train - num_val

    if split_strategy == 1:
        generator = torch.Generator().manual_seed(seed)
        x_train, x_val, x_test = random_split(
            line_indices, (num_train, num_val, num_test), generator
        )
    else:
        x_train = line_indices[:num_train]
        x_val = line_indices[num_train : num_train + num_val]
        x_test = line_indices[num_train + num_val :]

    return {
        "train": x_train,
        "test": x_test,
        "val": x_val,
        "feat_mapper": feat_mapper,
        "defaults": defaults,
        "metadata": metadata,
    }


class _AvazuSingletonOnRam(ICTRDataset):
    def __init__(self, txt_path: str, line_info: str, cache_path: Optional[str] = None):
        self._info = torch.load(line_info)
        self._feat_mapper = merge_feat_mapper_default(
            self._info["feat_mapper"], self._info["defaults"]
        )

        # Load full data to disk
        logger.debug("loading data to RAM...")
        self._load_data(txt_path)
        logger.debug("loaded data to RAM")

        self._num_train = len(self._info["train"])
        self._num_val = len(self._info["val"])
        self._num_test = len(self._info["test"])

        if cache_path:
            logger.debug(f"saved data to {cache_path}")
            torch.save(self.data, cache_path)

    def _get_subset(
        self, name: Literal["train", "val", "test"]
    ) -> Subset[Tuple[torch.Tensor, float]]:
        # get indices (train -> val -> test)
        if name == "train":
            indices = list(range(self._num_train))
        elif name == "val":
            indices = list(range(self._num_train, self._num_train + self._num_val))
        elif name == "val":
            indices = list(range(self._num_train + self._num_val, len(self)))
        else:
            raise ValueError(f"{name=}. Expected name in [train, val, test]")

        return Subset(self, indices)

    def _load_data(self, txt_path):
        # data: List[Optional[Tuple[torch.Tensor, float]]] = []
        data: Dict[int, Tuple[torch.Tensor, float]] = {}

        with open(txt_path) as fin:
            fin.readline()  # skip header
            pbar = tqdm(enumerate(fin), mininterval=1, smoothing=0.1)
            for idx, line in pbar:
                values = line.rstrip("\n").split(",")
                if len(values) != NUM_FEATS + 2:
                    # data.append(None)
                    continue

                feats = []
                for feat_idx, feat in enumerate(values[2:]):
                    # feat_id count from 1
                    feat_id = feat_idx + 1
                    feat_value = self._feat_mapper[feat_id][feat]
                    feats.append(feat_value)

                label = int(values[1])
                data[idx] = (torch.tensor(feats), label)
        self.data = data

    def __getitem__(self, idx):
        true_idx: int
        if idx <= self._num_train:
            true_idx = self._info["train"][idx]
        else:
            idx -= self._num_train
            if idx <= self._num_val:
                true_idx = self._info["val"][idx]
            else:
                idx -= self._num_val
                assert idx > 0
                true_idx = self._info["test"][idx]

        return self.data[true_idx]

    def __len__(self):
        return self._num_train + self._num_val + self._num_test

    def pop_info(self):
        return {}

    def describe(self):
        return


@lru_cache(1)
def _get_avazu_on_ram(
    txt_path: str, line_info: str, cache_path: Optional[str] = None
) -> _AvazuSingletonOnRam:
    return _AvazuSingletonOnRam(txt_path, line_info, cache_path)


class AvazuOnRam(ICTRDataset):
    def __init__(
        self,
        txt_path: str,
        line_info: str,
        name: Literal["train", "val", "test"],
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            txt_path: Path to csv file of dataset
            line_info: Path to binary file contains what line in the dataset
            name
            cache_path: Path to cache data
        """
        assert name in [
            "train",
            "val",
            "test",
        ], f"{name} must be in ['train', 'val', 'test']"
        full_data = _get_avazu_on_ram(txt_path, line_info, cache_path)
        self._dataset = full_data._get_subset(name)
        self.__getitems__ = self._dataset.__getitems__

    def pop_info(self):
        return {}

    def describe(self):
        return

    def __getitem__(self, idx: int):
        return self._dataset[idx]


if __name__ == "__main__":
    txt_path = "/home/hungt/tmp/train"
    line_info = "dataset/ctr/avazu/train_test_info.bin"

    binary = _create_binary(txt_path)
    print("Total data", sum(len(binary[k]) for k in ["train", "val", "test"]))
    torch.save(binary, line_info)
