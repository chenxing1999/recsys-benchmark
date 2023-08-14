import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Final, Set

NUM_INT_FEATS: Final[int] = 13
NUM_FEATS: Final[int] = 39

# feat_mapper[FeatureIndex][FeatureValue] = FeatureId
FeatMapper = Dict[int, Dict[str, int]]


def get_cache_data(
    path: str,
    min_threshold: int = 10,
    save_line=False,
) -> Dict[str, Any]:
    """Get cache object

    Args:
        path: Path to original train.txt / test.txt file
            of CriteoDataset

        min_threshold

        save_line

    Returns: Dict[str, Any]

        feat_mappers: (FeatMapper)
            feat_mapper[feat_idx][feat_value] = feat_id corresponding to feature_idx

        defaults (Dict[int, int])
            defaults[feat_idx] = default value for OOV feature of feat_idx

        num_data (int)
        line_idx_to_byte_loc: List[int]:
    """

    feat_cnts: DefaultDict[int, DefaultDict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    counts = 0
    line_idx_to_byte_loc = []
    with open(path) as fin:
        loc = 0
        for line in fin:
            loc += len(line)
            values = line.rstrip("\n").split("\t")
            if len(values) != NUM_FEATS + 1:
                continue

            for i in range(1, NUM_INT_FEATS + 1):
                feat_cnts[i][convert_numeric_feature(values[i])] += 1
            for i in range(NUM_INT_FEATS + 1, NUM_FEATS + 1):
                feat_cnts[i][values[i]] += 1

            if save_line:
                line_idx_to_byte_loc.append(loc - len(line))
            counts += 1

    convert_numeric_feature.cache_clear()
    # feat_idx_to_set map from FeatureIndex to Set of feature values
    feat_idx_to_set: Dict[int, Set[int]] = {
        i: {feat for feat, c in cnt.items() if c >= min_threshold}
        for i, cnt in feat_cnts.items()
    }

    feat_mappers: Dict[int, Dict[str, int]] = {
        i: {feat: idx for idx, feat in enumerate(cnt)}
        for i, cnt in feat_idx_to_set.items()
    }
    defaults = {i: len(cnt) for i, cnt in feat_mappers.items()}

    info = {
        "feat_mappers": feat_mappers,
        "defaults": defaults,
        "num_data": counts,
    }

    if save_line:
        info["line_idx_to_byte_loc"] = line_idx_to_byte_loc
    return info


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str) -> str:
    if val == "":
        return "NULL"
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)


def merge_feat_mapper_default(
    feat_mappers, defaults
) -> Dict[int, DefaultDict[str, int]]:
    feat_mappers: Dict[int, DefaultDict[str, int]] = {}
    for i, values in feat_mappers.items():
        feat_mappers[i] = defaultdict(lambda: defaults[i])
        feat_mappers[i].update(values)
    return feat_mappers
