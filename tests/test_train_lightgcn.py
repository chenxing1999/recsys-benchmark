import os

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.models.lightgcn import LightGCN
from src.trainer.lightgcn import train_epoch, validate_epoch

CUR_DIR = os.path.dirname(__file__)
SAMPLE_DATASET = os.path.join(CUR_DIR, "assets/sample_cf.txt")


@pytest.fixture
def train_dataset() -> CFGraphDataset:
    dataset = CFGraphDataset(SAMPLE_DATASET)
    assert dataset.num_items == 102
    assert dataset.num_users == 77
    return dataset


@pytest.fixture
def val_dataset() -> TestCFGraphDataset:
    dataset = TestCFGraphDataset(SAMPLE_DATASET)
    return dataset


@pytest.fixture
def model(train_dataset: CFGraphDataset) -> LightGCN:
    return LightGCN(train_dataset.num_users, train_dataset.num_items)


def test_train_simple(train_dataset, model):
    loader = DataLoader(train_dataset, batch_size=24, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.Adam(model.parameters())
    loss_dict = train_epoch(
        loader,
        model,
        optimizer,
        device,
        weight_decay=1e-5,
        info_nce_weight=0.1,
    )

    for loss in loss_dict.values():
        assert loss > 0
        assert isinstance(loss, float)


def test_train_no_info_nce_no_decay(train_dataset, model):
    loader = DataLoader(train_dataset, batch_size=24, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.Adam(model.parameters())
    loss_dict = train_epoch(
        loader,
        model,
        optimizer,
        device,
        weight_decay=0.0,
        info_nce_weight=0.0,
    )

    assert abs(loss_dict["rec_loss"] - loss_dict["loss"]) < 1e-8


def test_val_simple(train_dataset, val_dataset, model):
    loader = DataLoader(
        val_dataset,
        batch_size=24,
        num_workers=2,
        collate_fn=val_dataset.collate_fn,
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metrics = validate_epoch(
        train_dataset,
        loader,
        model,
        device,
    )

    # all correct result already in train --> filtered --> ndcg is 0
    assert metrics["ndcg"] == 0
