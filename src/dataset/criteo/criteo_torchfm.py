"""based on TorchFm Criteo implementation"""
import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..base import ICTRDataset


class CriteoDataset(ICTRDataset):
    """
    Criteo Display Advertising Challenge Dataset

    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances)
            and treat them as a single feature
        * Discretize numerical values by log2 transformation which is
            proposed by the winner of Criteo Competition

    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.

    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(
        self,
        train_test_info,
        dataset_name,
        my_path=None,
        dataset_path=None,
        cache_path=".criteo",
        rebuild_cache=False,
        min_threshold=10,
    ):
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError("create cache: failed: dataset_path is None")
            self.__build_cache(dataset_path, cache_path, my_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)

        # hook my dataset
        self._line_in_dataset = list(torch.load(train_test_info)[dataset_name])
        self._line_in_dataset.sort()

        with self.env.begin(write=False) as txn:
            self.field_dims = np.frombuffer(
                txn.get(b"field_dims"), dtype=np.uint32
            ).astype(np.int64)

    def __getitem__(self, index):
        index = self._line_in_dataset[index]
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack(">I", index)), dtype=np.uint32
            ).astype(dtype=np.int64)
        return np_array[1:], np_array[0]

    def __len__(self):
        return len(self._line_in_dataset)

    def __build_cache(self, path, cache_path, my_path=None):
        if my_path:
            feat_mapper, defaults = self.set_featmappers(my_path)
        else:
            feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b"field_dims", field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description("Create criteo dataset cache: counting features")
            for line in pbar:
                values = line.rstrip("\n").split("\t")
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
        feat_mapper = {
            i: {feat for feat, c in cnt.items() if c >= self.min_threshold}
            for i, cnt in feat_cnts.items()
        }
        feat_mapper = {
            i: {feat: idx for idx, feat in enumerate(cnt)}
            for i, cnt in feat_mapper.items()
        }
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description("Create criteo dataset cache: setup lmdb")
            for item_idx, line in enumerate(pbar):
                values = line.rstrip("\n").split("\t")
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(
                        convert_numeric_feature(values[i]), defaults[i]
                    )
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack(">I", item_idx), np_array.tobytes()))
                # item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

    def pop_info(self):
        return {}

    def set_featmappers(self, my_path):
        data = torch.load(my_path)
        feat_mappers = data["feat_mappers"]
        defaults = data["defaults"]

        return feat_mappers, defaults

    def describe(self):
        logger.info("Normal Criteo Dataset")
        logger.info(f"Num data: {len(self)}")
        logger.info(f"Field dims {self.field_dims}")
        logger.info(f"sum dims {sum(self.field_dims)}")


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == "":
        return "NULL"
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)
