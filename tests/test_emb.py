import os
import tempfile
from typing import Tuple

import pytest
import torch

from src.models.embeddings import (
    NAME_TO_CLS,
    DHEmbedding,
    QRHashingEmbedding,
    get_embedding,
)
from src.models.embeddings.tensortrain_embeddings import TT_EMB_AVAILABLE

numba_cuda_available = False
numba_available = False
try:
    import numba  # noqa: F401

    numba_available = True
    numba_cuda_available = torch.cuda.is_available()
except ImportError:
    pass


def create_sparse_tensor(
    sparse_rate: float, size: Tuple[int, int], device="cuda"
) -> torch.Tensor:
    """Create random sparse tensor

    Args:
        sparse_rate
        size
    """
    num_values = int(size[0] * size[1] * (1 - sparse_rate))
    v = torch.ones(num_values, device=device)
    ind = torch.stack(
        [
            torch.randint(size[0], size=(num_values,), device=device),
            torch.randint(size[1], size=(num_values,), device=device),
        ]
    )
    return torch.sparse_coo_tensor(
        ind,
        v,
        size=size,
    )


@pytest.mark.parametrize(
    ["num_item", "divider", "expected_size"],
    [
        [32, 8, 4],
        [31, 8, 4],
        [33, 8, 5],
        [1, 8, 1],
    ],
)
def test_init_logic_qr(num_item, divider, expected_size):
    QRHashingEmbedding(num_item, 16, divider).emb2.num_embeddings == expected_size


@pytest.mark.parametrize(
    ["num_item", "inp_size"],
    [
        [32, 64],
        [13, 32],
    ],
)
def test_init_logic_dhe(num_item, inp_size):
    ORIGINAL_DHE_COUNTER = DHEmbedding.COUNTER
    emb1 = DHEmbedding(num_item, 64, None, inp_size, [64])

    # Reset counter to create same embedding cache
    DHEmbedding.COUNTER = ORIGINAL_DHE_COUNTER
    emb2 = DHEmbedding(num_item, 64, None, inp_size, [64])

    # Reset counter to reset state change
    DHEmbedding.COUNTER = ORIGINAL_DHE_COUNTER

    cache1 = emb1._cache
    cache2 = emb2._cache
    assert len(cache1) == len(cache2)
    assert len(cache1) == num_item

    for v1, v2 in zip(cache1, cache2):
        assert (v1 == v2).all()


@pytest.mark.parametrize(
    ["num_item", "inp_size"],
    [
        [32, 64],
        [13, 32],
    ],
)
def test_init_dhe_twice(num_item, inp_size):
    emb1 = DHEmbedding(num_item, 64, None, inp_size, [64])

    emb2 = DHEmbedding(num_item, 64, None, inp_size, [64])

    cache1 = emb1._cache
    cache2 = emb2._cache
    assert len(cache1) == len(cache2)
    assert len(cache1) == num_item

    for v1, v2 in zip(cache1, cache2):
        assert (v1 != v2).any()


EMBEDDING_NAMES = list(NAME_TO_CLS.keys())
GENERAL_EMB = [
    name
    for name in EMBEDDING_NAMES
    if (
        not name.startswith("pep")
        and not name.startswith("deepfm")
        and name != "cerp_retrain"
        and name != "tt_emb"
    )
]


@pytest.mark.parametrize("name", GENERAL_EMB)
def test_api_embedding_lightgcn(name: str):
    num_item = 3
    hidden_size = 7
    batch_size = 11

    emb_config = {"name": name}
    emb = get_embedding(emb_config, num_item, hidden_size)
    inp = torch.randint(num_item, size=(batch_size,))
    res = emb(inp)

    assert res.shape == (batch_size, hidden_size)


@pytest.mark.parametrize("name", GENERAL_EMB)
def test_api_embedding_lightgcn_get_weight(name: str):
    num_item = 3
    hidden_size = 7

    emb_config = {"name": name}
    emb = get_embedding(emb_config, num_item, hidden_size)
    res = emb.get_weight()

    assert res.shape == (num_item, hidden_size)


@pytest.mark.parametrize("name", GENERAL_EMB + ["deepfm_optembed"])
def test_api_embedding_deepfm(name: str):
    field_dims = [2, 3, 5]
    hidden_size = 7
    batch_size = 11

    num_field = len(field_dims)
    num_item = sum(field_dims)

    emb_config = {"name": name}
    emb = get_embedding(emb_config, field_dims, hidden_size)
    inp = torch.randint(num_item, size=(batch_size, num_field))
    res = emb(inp)

    assert res.shape == (batch_size, num_field, hidden_size)


def test_api_embedding_pep():
    num_item = 3
    hidden_size = 7
    batch_size = 11

    emb_config = {"name": "pep"}
    # Pep have some weird API :(
    with tempfile.TemporaryDirectory() as tmpdir:
        emb_config["checkpoint_weight_dir"] = tmpdir
        emb = get_embedding(emb_config, num_item, hidden_size)
        inp = torch.randint(num_item, size=(batch_size,))
        res = emb(inp)
        assert res.shape == (batch_size, hidden_size)


def test_api_embedding_pep_retrain():
    num_item = 3
    hidden_size = 5
    batch_size = 11

    weight = [
        [0.2, 0.3, 0.4, 0.5, 0.5],
        [1.2, 1.3, 1.4, 1.5, 1.5],
        [0.2, 1.3, 0.4, 1.5, 1.5],
    ]
    weight = torch.tensor(weight)
    s = [
        [0.4, 0.2, 0.3, 0.9, 0.4],
        [1.1, 1.5, 1.1, 1.1, 1.8],
        [0.1, 1.1, 0.6, 1.2, 1.1],
    ]
    s = torch.tensor(s)
    tmp_state = {
        "emb.weight": weight,
        "s": s,
    }

    emb_config = {"name": "pep_retrain", "sparsity": "tmp"}
    with tempfile.TemporaryDirectory() as tmpdir:
        emb_config["checkpoint_weight_dir"] = tmpdir
        path = os.path.join(tmpdir, "tmp.pth")

        torch.save(tmp_state, path)

        emb = get_embedding(emb_config, num_item, hidden_size)
        inp = torch.randint(num_item, size=(batch_size,))
        res = emb(inp)
        assert res.shape == (batch_size, hidden_size)


@pytest.mark.parametrize("name", GENERAL_EMB)
def test_api_embedding_bag(name: str):
    num_item = 3
    hidden_size = 7
    batch_size = 11
    num_fields = 4
    mode = "mean"

    emb_config = {"name": name}
    emb = get_embedding(emb_config, num_item, hidden_size, mode)
    inp = torch.randint(num_item, size=(batch_size, num_fields))
    res = emb(inp)

    assert res.shape == (batch_size, hidden_size)


@pytest.mark.skipif(not TT_EMB_AVAILABLE, reason="TT Embedding is not available")
def test_tt_emb_api_lightgcn():
    num_item = 3
    hidden_size = 8

    emb_config = {
        "name": "tt_emb",
        "tt_ranks": [2, 4, 2],
    }
    emb = get_embedding(emb_config, num_item, hidden_size)
    res = emb.get_weight()

    assert res.shape == (num_item, hidden_size)


@pytest.mark.skipif(not TT_EMB_AVAILABLE, reason="TT Embedding is not available")
def test_tt_emb_result_lightgcn():
    num_item = 3
    hidden_size = 8

    emb_config = {
        "name": "tt_emb",
        "tt_ranks": [2, 4, 2],
    }
    inp = torch.tensor([1, 1, 2, 2], device="cuda")
    emb = get_embedding(emb_config, num_item, hidden_size)
    emb = emb.to("cuda")
    res = emb(inp)

    assert res.shape == (4, hidden_size)
    assert (res[0] == res[1]).all()
    assert (res[2] == res[3]).all()

    w = emb.get_weight()
    assert (w[1] == res[0]).all()


@pytest.mark.skipif(not TT_EMB_AVAILABLE, reason="TT Embedding is not available")
def test_tt_emb_result_deepfm():
    num_item = 3
    hidden_size = 8

    emb_config = {
        "name": "tt_emb",
        "tt_ranks": [2, 4, 2],
    }
    inp = torch.tensor([[1, 2, 0], [1, 2, 0]], device="cuda")
    emb = get_embedding(emb_config, num_item, hidden_size)
    emb = emb.to("cuda")
    res = emb(inp)

    assert res.shape == (2, 3, hidden_size)
    assert (res[0] == res[1]).all()


@pytest.mark.skipif(not TT_EMB_AVAILABLE, reason="TT Embedding is not available")
def test_tt_emb_assertion_device_model():
    num_item = 3
    hidden_size = 8

    emb_config = {
        "name": "tt_emb",
        "tt_ranks": [2, 4, 2],
    }
    inp = torch.tensor([1, 1, 2, 2], device="cuda")
    emb = get_embedding(emb_config, num_item, hidden_size)

    with pytest.raises(AssertionError):
        emb(inp)


@pytest.mark.skipif(not TT_EMB_AVAILABLE, reason="TT Embedding is not available")
def test_tt_emb_assertion_device_input():
    num_item = 3
    hidden_size = 8

    emb_config = {
        "name": "tt_emb",
        "tt_ranks": [2, 4, 2],
    }
    inp = torch.tensor([1, 1, 2, 2], device="cpu")
    emb = get_embedding(emb_config, num_item, hidden_size)
    emb = emb.to("cuda")

    with pytest.raises(AssertionError):
        emb(inp)


@pytest.mark.skipif(not numba_cuda_available, reason="Numba or CUDA is not available")
def test_pruned_emb_cuda():
    from src.models.embeddings.base import VanillaEmbedding
    from src.models.embeddings.pruned_embedding import PrunedEmbedding

    field_dims = [512, 512]
    num_items = sum(field_dims)
    model = VanillaEmbedding(field_dims, 16)
    weight = create_sparse_tensor(0.9, (num_items, 16), "cpu").to_dense()
    model._emb_module.weight.data = weight

    pruned_embedding = PrunedEmbedding.from_other_emb(model)

    model.cuda()
    pruned_embedding.to_cuda()

    inp = torch.randint(num_items, size=(256,), device="cuda")
    results = pruned_embedding(inp)
    expected = model(inp)

    assert results.isclose(expected).all()
    assert pruned_embedding.get_weight().isclose(weight).all()


@pytest.mark.skipif(not numba_available, reason="Numba is not available")
def test_pruned_emb_cpu():
    from src.models.embeddings.base import VanillaEmbedding
    from src.models.embeddings.pruned_embedding import PrunedEmbedding

    field_dims = [512, 512]
    num_items = sum(field_dims)
    model = VanillaEmbedding(field_dims, 16)
    weight = create_sparse_tensor(0.9, (num_items, 16), "cpu").to_dense()
    model._emb_module.weight.data = weight

    pruned_embedding = PrunedEmbedding.from_other_emb(model)

    inp = torch.randint(num_items, size=(256,))
    results = pruned_embedding(inp)
    expected = model(inp)

    assert results.isclose(expected).all()
    assert pruned_embedding.get_weight().isclose(weight).all()


def test_dhe_without_cache_inference():
    ORIGINAL_DHE_COUNTER = DHEmbedding.COUNTER
    emb_with_cache = DHEmbedding(1024, 16, hidden_sizes=[32, 32])
    emb_with_cache.eval()

    # Reset counter to create same embedding cache
    DHEmbedding.COUNTER = ORIGINAL_DHE_COUNTER
    emb_without_cache = DHEmbedding(1024, 16, hidden_sizes=[32, 32], cached=False)
    emb_without_cache.eval()

    # Reset counter to reset state change
    DHEmbedding.COUNTER = ORIGINAL_DHE_COUNTER

    emb_without_cache._seq.load_state_dict(emb_with_cache._seq.state_dict())

    inp = torch.randint(1024, size=(32,))

    with torch.no_grad():
        assert emb_with_cache(inp).isclose(emb_without_cache(inp)).all()
