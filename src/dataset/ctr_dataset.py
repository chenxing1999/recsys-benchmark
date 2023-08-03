"""CTR Dataset based on Dive into Deep Learning implementation """

from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    field_dims: np.ndarray

    def __init__(
        self, data_path, feat_mapper=None, defaults=None, min_threshold=4, num_feat=34
    ):
        """
        Args:
            data_path: Path to tsv data
            feat_mapper (Optional[Dict[int, Dict[feature, index]]])
                key is index of feature (Column in DataFrame)
                    `feature` is exact value of feature
                    `index`is the actual value / ids

            defaults (Optional[Dict[int, index]])

            min_threshold: Assign value to default
                if it appears less than min_threshold times
            num_feat: Number of features used
        """
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}

        # feat_cnts[i][j]: Number appearance of feature type i with value j
        feat_cnts = defaultdict(lambda: defaultdict(int))

        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip("\n").split("\t")
                if len(values) != self.NUM_FEATS + 1:
                    continue

                label = int(values[0])
                assert label in [0, 1]
                instance["y"] = label

                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault("x", []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1

        # Init feat_mapper and defaults
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {
                i: {feat for feat, c in cnt.items() if c >= min_threshold}
                for i, cnt in feat_cnts.items()
            }
            self.feat_mapper = {
                i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                for i, feat_values in feat_mapper.items()
            }
            self.defaults = {
                i: len(feat_values) for i, feat_values in feat_mapper.items()
            }

        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1

        self.offsets = np.array((0, *np.cumsum(self.field_dims).tolist()[:-1]))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = np.array(
            [
                self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                for i, v in enumerate(self.data[idx]["x"])
            ]
        )
        return feat + self.offsets, self.data[idx]["y"]
