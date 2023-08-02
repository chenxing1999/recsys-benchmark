import pytest

from src.models.embeddings import DHEEmbedding, QRHashingEmbedding


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
    emb1 = DHEEmbedding(num_item, 64, inp_size, [64])

    emb2 = DHEEmbedding(num_item, 64, inp_size, [64])

    cache1 = emb1._cache
    cache2 = emb2._cache
    assert len(cache1) == len(cache2)
    assert len(cache1) == num_item

    for v1, v2 in zip(cache1, cache2):
        assert (v1 == v2).all()
