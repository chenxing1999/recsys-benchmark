import os
from collections import defaultdict
from typing import DefaultDict, Dict, Final, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset

from .base import ICriteoDatset
from .utils import convert_numeric_feature, get_cache_data

# feat_mapper[FeatureIndex][FeatureValue] = FeatureId
FeatMapper = Dict[int, Dict[str, int]]


# Construct feat mapper and default
class CriteoDataset(Dataset, ICriteoDatset):
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

        if not os.path.exists(cache_path):
            logger.info("Creating cache data...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cached_data = get_cache_data(path, min_threshold, save_line=True)

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
        self.line_idx_to_byte = cached_data["line_idx_to_byte_loc"]
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

        loc = self.line_idx_to_byte[idx]
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
        logger.info("Normal Criteo Dataset")
        logger.info("Num data:", self.num_data)
        logger.info("Field dims", self.field_dims)


if __name__ == "__main__":
    dataset = CriteoDataset("", ".criteo/train.bin")