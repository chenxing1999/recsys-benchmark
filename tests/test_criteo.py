import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import get_ctr_dataset
from src.dataset.criteo import CriteoDataset, CriteoFMDataset, CriteoIterDataset
from src.dataset.criteo.utils import NUM_FEATS

CUR_DIR = os.path.dirname(__file__)
SAMPLE_DATASET = os.path.join(CUR_DIR, "assets/train_criteo_sample.txt")


@pytest.fixture
def storage_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        return tmp_dir


@pytest.fixture
def cache_path(storage_dir):
    return os.path.join(storage_dir, "cache.bin")


# Required setup for below test
@pytest.fixture(autouse=True)
def test_get_criteo_simple(cache_path):
    dataloader_config = dict(
        dataset=dict(
            path=SAMPLE_DATASET,
            cache_path=cache_path,
        ),
        num_workers=2,
        batch_size=4,
    )

    dataset = get_ctr_dataset(dataloader_config)
    assert isinstance(dataset, CriteoDataset)
    assert len(dataset) == 100
    assert len(dataset.feat_mappers[5]) == 0
    assert os.path.exists(cache_path)


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


def test_get_criteo_fm():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_test_info = {
            "train": list(range(80)),
            "val": list(range(80, 90)),
            "test": list(range(90, 100)),
        }

        train_test_info_path = os.path.join(tmp_dir, "train_test_info.bin")
        torch.save(train_test_info, train_test_info_path)
        cache_path = os.path.join(tmp_dir, "cache_fm/")
        dataloader_config = dict(
            dataset=dict(
                dataset_path=SAMPLE_DATASET,
                cache_path=cache_path,
                dataset_name="train",
                train_test_info=train_test_info_path,
            ),
            num_workers=2,
            batch_size=4,
        )

        dataset = get_ctr_dataset(dataloader_config)
        assert isinstance(dataset, CriteoFMDataset)
        assert len(dataset) == 80


def test_get_criteo_fm_with_cache(cache_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # sorry for bad naming scheme
        my_path = cache_path

        train_test_info = {
            "train": list(range(80)),
            "val": list(range(80, 90)),
            "test": list(range(90, 100)),
        }

        train_test_info_path = os.path.join(tmp_dir, "train_test_info.bin")
        torch.save(train_test_info, train_test_info_path)
        cache_path = os.path.join(tmp_dir, "cache_fm/")
        dataloader_config = dict(
            dataset=dict(
                dataset_path=SAMPLE_DATASET,
                cache_path=cache_path,
                my_path=my_path,
                dataset_name="train",
                train_test_info=train_test_info_path,
            ),
            num_workers=2,
            batch_size=4,
        )

        dataset = get_ctr_dataset(dataloader_config)
        assert isinstance(dataset, CriteoFMDataset)
        assert len(dataset) == 80


def test_get_criteo_fm_get(cache_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # sorry for bad naming scheme
        my_path = cache_path

        train_test_info = {
            "train": list(range(80)),
            "val": list(range(80, 90)),
            "test": list(range(90, 100)),
        }

        train_test_info_path = os.path.join(tmp_dir, "train_test_info.bin")
        torch.save(train_test_info, train_test_info_path)
        cache_path = os.path.join(tmp_dir, "cache_fm/")
        dataloader_config = dict(
            dataset=dict(
                dataset_path=SAMPLE_DATASET,
                cache_path=cache_path,
                my_path=my_path,
                dataset_name="train",
                train_test_info=train_test_info_path,
            ),
            num_workers=2,
            batch_size=4,
        )

        dataset = get_ctr_dataset(dataloader_config)
        assert isinstance(dataset, CriteoFMDataset)
        assert len(dataset) == 80

        indices = list(range(30, 50))
        x, y = torch.utils.data.default_collate([dataset[idx] for idx in indices])

        x2, y2 = torch.utils.data.default_collate(dataset.__getitems__(indices))

        assert (x == x2).all()
        assert (y == y2).all()

        # Sanity check with loader
        loader = DataLoader(dataset, 50)

        batch_x, batch_y = next(iter(loader))
        assert (batch_x[30:] == x).all()
        assert (batch_y[30:] == y).all()
