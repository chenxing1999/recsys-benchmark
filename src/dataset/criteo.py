import math
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Final, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

# feat_mapper[FeatureIndex][FeatureValue] = FeatureId
FeatMapper = Dict[int, Dict[str, int]]


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str) -> str:
    if val == "":
        return "NULL"
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)


# Construct feat mapper and default
class CriteoDataset(Dataset):
    """
    Note: This implementation is based on pytorch-fm original implementation.
        The main difference is removing lmdb and changing reading logic.
        On my laptop, this improve training speed by around 10 times.
    """

    NUM_INT_FEATS: Final[int] = 13
    NUM_FEATS: Final[int] = 39

    def __init__(
        self,
        path: str,
        cache_path: str,
        min_threshold=10,
        feat_mappers: Optional[FeatMapper] = None,
        defaults: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            path (str): Path to the orginal train.txt / test.txt

            cache_path (str): Path to cached / preprocessed information
                if available, load this instead of path
                if not available, create this from path

            min_threshold (int): If less than min_threshold, the value will
                be set to OOV

            feat_mappers: If not None, always use this instead of from `cache` or `path`
            defaults: If not None, always use this instead of from `cache` or `path`
        """
        self.min_threshold = min_threshold

        if not os.path.exists(cache_path):
            logger.info("Creating cache data...")
            cached_data = self._get_cache_data(path)

            if feat_mappers:
                cached_data["feat_mappers"] = feat_mappers

            if defaults:
                cached_data["defaults"] = defaults

            torch.save(cached_data, cache_path)
            logger.info("Done")
        else:
            cached_data = torch.load(cache_path)

        if feat_mappers is None:
            feat_mappers = cached_data["feat_mappers"]
        if defaults is None:
            defaults = cached_data["defaults"]
        self.num_data = cached_data["num_data"]

        # Begin hacking
        # line_idx_to_byte = cached_data["line_idx_to_byte_loc"]
        # arr = np.array(list(line_idx_to_byte.values()), dtype=np.uint64)
        # mem = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        # b = np.ndarray(arr.shape, dtype=arr.dtype, buffer=mem.buf)
        # b[:] = arr[:]
        # self.name = mem.name
        # del line_idx_to_byte
        # mem.close()
        line_idx_to_byte = cached_data["line_idx_to_byte_loc"]
        self.arr = np.array(list(line_idx_to_byte.values()), dtype=np.uint64)
        del line_idx_to_byte
        # end hacking

        self._defaults = defaults
        self._feat_mappers = feat_mappers
        # Construct actual feat mapper dictionary
        self.feat_mappers: Dict[int, DefaultDict[str, int]] = {}
        for i, values in feat_mappers.items():
            self.feat_mappers[i] = defaultdict(lambda: defaults[i])
            self.feat_mappers[i].update(values)

        self.field_dims = [
            len(self.feat_mappers[i + 1]) + 1 for i in range(self.NUM_FEATS)
        ]

        self.path = path

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feat_mapper = self.feat_mappers

        loc = self.arr[idx]
        with open(self.path) as fin:
            fin.seek(loc)
            line = fin.readline()

        line = line.rstrip("\n").split("\t")
        label = int(line[0])

        feats = [0] * self.NUM_FEATS
        for i in range(1, self.NUM_INT_FEATS + 1):
            value = convert_numeric_feature(line[i])
            feats[i - 1] = feat_mapper[i][value]

        for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
            feats[i - 1] = feat_mapper[i][line[i]]

        return torch.tensor(feats), label

    def _get_cache_data(self, path) -> Dict[str, Any]:
        """Get cache object

        Args:
            path: Path to original train.txt / test.txt file
                of CriteoDataset

        Returns: Dict[str, Any]

            feat_mappers: (FeatMapper)
                feat_mapper[feat_idx][feat_value] = feat_id corresponding to feature_idx

            defaults (Dict[int, int])
                defaults[feat_idx] = default value for OOV feature of feat_idx

            num_data (int)
            line_idx_to_byte_loc: Dict[int, int]:
        """

        feat_cnts: DefaultDict[int, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        counts = 0
        line_idx_to_byte_loc = {}
        with open(path) as fin:
            loc = 0
            for line in fin:
                loc += len(line)
                values = line.rstrip("\n").split("\t")
                if len(values) != self.NUM_FEATS + 1:
                    continue

                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1

                line_idx_to_byte_loc[counts] = loc - len(line)
                counts += 1

        # feat_idx_to_set map from FeatureIndex to Set of feature values
        feat_idx_to_set: Dict[int, Set[int]] = {
            i: {feat for feat, c in cnt.items() if c >= self.min_threshold}
            for i, cnt in feat_cnts.items()
        }

        feat_mappers: Dict[int, Dict[str, int]] = {
            i: {feat: idx for idx, feat in enumerate(cnt)}
            for i, cnt in feat_idx_to_set.items()
        }
        defaults = {i: len(cnt) for i, cnt in feat_mappers.items()}

        return {
            "feat_mappers": feat_mappers,
            "defaults": defaults,
            "line_idx_to_byte_loc": line_idx_to_byte_loc,
            "num_data": counts,
        }

    def pop_info(self):
        # TODO: Refactor this
        feat_mappers = self._feat_mappers
        defaults = self._defaults
        self._feat_mappers = None
        self._defaults = None
        return {
            "feat_mappers": feat_mappers,
            "defaults": defaults,
        }

    def describe(self):
        print("Num data:", self.num_data)


if __name__ == "__main__":
    dataset = CriteoDataset("", ".criteo/train.bin")
