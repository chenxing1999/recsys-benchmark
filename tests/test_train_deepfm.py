import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset.criteo import CriteoDataset
from src.models.deepfm import DeepFM, get_optimizers
from src.trainer.deepfm import train_epoch, train_epoch_cerp, validate_epoch

CUR_DIR = os.path.dirname(__file__)
SAMPLE_DATASET = os.path.join(CUR_DIR, "assets/train_criteo_sample.txt")


@pytest.fixture
def dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = os.path.join(tmp_dir, "cache.bin")
        return CriteoDataset(SAMPLE_DATASET, cache_path)


@pytest.fixture
def model(dataset):
    return DeepFM(dataset.field_dims, 12, [8, 12])


def test_train_simple(dataset, model):
    loader = DataLoader(dataset, batch_size=24, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.Adam(model.parameters())
    loss_dict = train_epoch(loader, model, optimizer, device)
    assert loss_dict["loss"] > 0
    assert isinstance(loss_dict["loss"], float)


def test_train_cerp(dataset):
    cerp_config = dict(
        name="cerp",
        bucket_size=120000,
        threshold_init=-100,
    )

    model = DeepFM(dataset.field_dims, 12, [8, 12], embedding_config=cerp_config)
    loader = DataLoader(dataset, batch_size=24, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.Adam(model.parameters())
    loss_dict = train_epoch_cerp(
        loader,
        model,
        optimizer,
        device,
        prune_loss_weight=1,
    )
    assert loss_dict["log_loss"] > 0
    assert isinstance(loss_dict["loss"], float)


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_train_sparse(dataset, optimizer_name):
    loader = DataLoader(dataset, batch_size=24, num_workers=2)
    model = DeepFM(
        dataset.field_dims,
        12,
        [8, 12],
        embedding_config={"name": "vanilla", "sparse": True},
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    config = {
        "sparse": True,
        "name": "adam",
        "weight_decay": 1e-6,
        "learning_rate": 1e-3,
    }
    optimizers = get_optimizers(model, config)

    if optimizer_name == "adam":
        assert isinstance(optimizers, list) and len(optimizers) == 2
    loss_dict = train_epoch(loader, model, optimizers, device)
    assert loss_dict["loss"] > 0
    assert isinstance(loss_dict["loss"], float)


@pytest.mark.parametrize("name", ["sgd", "adam"])
@pytest.mark.parametrize("sparse", [True, False])
def test_get_optimizers(name, sparse, model):
    config = {
        "sparse": sparse,
        "name": name,
        "weight_decay": 1e-6,
        "learning_rate": 1e-3,
    }
    optimizers = get_optimizers(model, config)
    assert isinstance(optimizers, list)


def test_eval_simple(dataset, model):
    loader = DataLoader(dataset, batch_size=24, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metrics = validate_epoch(loader, model, device)

    assert metrics["log_loss"] > 0
    assert isinstance(metrics["log_loss"], float)

    assert metrics["auc"] >= 0 and metrics["auc"] <= 1
    assert isinstance(metrics["auc"], float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_optembed_evol(dataset):
    from src.models.embeddings.deepfm_opt_embed import evol_search_deepfm

    emb_config = {
        "name": "deepfm_optembed",
    }
    opt_deepfm_model = DeepFM(
        dataset.field_dims,
        8,
        [12, 12],
        0.1,
        False,
        emb_config,
    )

    loader = DataLoader(dataset, batch_size=24, num_workers=2)

    mask_d, auc = evol_search_deepfm(
        opt_deepfm_model,
        2,
        5,
        2,
        2,
        0.1,
        3,
        loader,
        dataset,
    )
    mask_e = opt_deepfm_model.embedding.get_mask_e()

    emb_config = {
        "name": "deepfm_optembed_retrain",
    }
    opt_deepfm_model = DeepFM(
        dataset.field_dims,
        8,
        [12, 12],
        0.1,
        False,
        emb_config,
    )
    opt_deepfm_model.embedding.init_mask(mask_e, mask_d)

    retrain_sparsity = opt_deepfm_model.embedding.get_sparsity()
    assert retrain_sparsity > 0
