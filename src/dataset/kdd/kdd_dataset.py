import shutil
import struct
from pathlib import Path
from typing import Literal

import lmdb
import numpy as np
import torch.utils.data
from loguru import logger
from tqdm import tqdm

from src.dataset.base import ICTRDataset


class KddDataset(ICTRDataset):
    """
    KDD Click-Through Rate Prediction Dataset
        https://www.kaggle.com/c/kddcup2012-track2
    """

    def __init__(
        self,
        train_test_info: str,
        dataset_name: Literal["train", "val", "test"],
        dataset_path=None,
        cache_path=".kdd",
        rebuild_cache=False,
        min_threshold=10,
    ):
        self.NUM_FEATS = 11
        self.min_threshold = min_threshold

        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError("create cache: failed: dataset_path is None")
            self.__build_cache(
                dataset_path,
                cache_path,
                train_test_info,
            )
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)

        train_test_info = torch.load(train_test_info)

        self._line_in_dataset = list(train_test_info[dataset_name])
        self._line_in_dataset.sort()
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1
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

    def __build_cache(
        self,
        path: str,
        cache_path: str,
        train_test_info,
    ):
        feat_mapper = train_test_info["feat_mapper"]
        defaults = train_test_info["defaults"]

        num_feats = self.NUM_FEATS

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(num_feats, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b"field_dims", field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __yield_buffer(self, path: str, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description("Create avazu dataset cache: setup lmdb")
            for line in pbar:
                values = line.rstrip("\n").split("\t")
                assert len(values) == self.NUM_FEATS + 1

                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])

                buffer.append((struct.pack(">I", item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

    def describe(self):
        logger.info("KDD dataset")
        logger.info(f"sum field dims: {sum(self.field_dims)}")
        logger.info(f"length field dims: {len(self.field_dims)}")
        return

    def pop_info(self):
        return {}


if __name__ == "__main__":
    import os

    dataset_path = "dataset/ctr/kdd/"
    train_test_info = os.path.join(dataset_path, "preprocessed/train_test_val_info.bin")
    dataset_path = os.path.join(dataset_path, "track2/traininng.txt")

    dataset = KddDataset(train_test_info, "train", dataset_path)
