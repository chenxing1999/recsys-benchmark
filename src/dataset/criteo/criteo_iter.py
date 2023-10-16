import os
from collections import defaultdict
from typing import DefaultDict, Dict, Optional

import torch
from loguru import logger
from torch.utils.data import IterableDataset

from .base import ICriteoDatset
from .utils import NUM_FEATS, NUM_INT_FEATS, convert_numeric_feature, get_cache_data

# feat_mapper[FeatureIndex][FeatureValue] = FeatureId
FeatMapper = Dict[int, Dict[str, int]]


class CriteoIterDataset(IterableDataset, ICriteoDatset):
    """
    Iterable version of src.dataset.criteo.criteo.CriteoDataset

    This version consume less memory and run slightly faster
        if your num_workers less than 4
    Note that this version doesn't allow you to shuffle dataset
        and set num_workers != 0
    """

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
            cached_data = get_cache_data(path, min_threshold, False)

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

        self._defaults = defaults
        self._feat_mappers = feat_mappers
        # Construct actual feat mapper dictionary
        self.feat_mappers: Dict[int, DefaultDict[str, int]] = {}
        for i, values in feat_mappers.items():
            self.feat_mappers[i] = defaultdict(lambda: defaults[i])
            self.feat_mappers[i].update(values)

        self.field_dims = [len(self.feat_mappers[i + 1]) + 1 for i in range(NUM_FEATS)]
        field_dims = torch.tensor(self.field_dims)
        field_dims = torch.cat([torch.tensor([0], dtype=torch.long), field_dims])
        self.offsets = torch.cumsum(field_dims[:-1], 0)

        self.path = path

    def __len__(self):
        return self.num_data

    def __iter__(self):
        return iter(self._get_generator())

    def _get_generator(self):
        feat_mapper = self.feat_mappers
        with open(self.path) as fin:
            for line in fin:
                line = line.rstrip("\n").split("\t")
                if len(line) != NUM_FEATS + 1:
                    continue

                label = int(line[0])

                feats = [0] * NUM_FEATS
                for i in range(1, NUM_INT_FEATS + 1):
                    value = convert_numeric_feature(line[i])
                    feats[i - 1] = feat_mapper[i][value]

                for i in range(NUM_INT_FEATS + 1, NUM_FEATS + 1):
                    feats[i - 1] = feat_mapper[i][line[i]]

                feats = torch.tensor(feats)
                # feats = feats + self.offsets
                yield feats, label

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
        logger.info("Iter Criteo Dataset")
        logger.info("Num data:", self.num_data)
