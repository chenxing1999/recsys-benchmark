import os
import tempfile

from src.dataset.criteo import CriteoDataset, CriteoIterDataset
from src.dataset.criteo.utils import NUM_FEATS

CUR_DIR = os.path.dirname(__file__)
SAMPLE_DATASET = os.path.join(CUR_DIR, "assets/train_criteo_sample.txt")


def test_criteo_dataset_simple():
    """Simple test for Criteo Dataset for syntax and simple logic error checking"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = os.path.join(tmp_dir, "cache.bin")
        dataset = CriteoDataset(SAMPLE_DATASET, cache_path)
        assert len(dataset) == 100
        assert len(dataset.feat_mappers[5]) == 0

        feature, label = dataset[0]
        assert label in [0, 1]
        assert len(feature) == NUM_FEATS
        assert len(dataset.field_dims) == NUM_FEATS

        feature = feature.tolist()

        for field_dim, feat in zip(dataset.field_dims, feature):
            assert feat < field_dim


def test_criteo_iter_dataset_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = os.path.join(tmp_dir, "cache.bin")
        dataset = CriteoIterDataset(SAMPLE_DATASET, cache_path)
        assert len(dataset.feat_mappers[5]) == 0

        iter_dataset = iter(dataset)
        feature, label = next(iter_dataset)
        assert label in [0, 1]
        assert len(feature) == NUM_FEATS
        assert len(dataset.field_dims) == NUM_FEATS

        feature = feature.tolist()

        for field_dim, feat in zip(dataset.field_dims, feature):
            assert feat < field_dim
