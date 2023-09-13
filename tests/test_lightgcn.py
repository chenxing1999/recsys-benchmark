from src.models.lightgcn import LightGCN


def test_lightgcn_dhe():
    num_user = 4
    num_item = 4

    embedding_config = {
        "name": "dhe",
    }

    model = LightGCN(num_user, num_item, embedding_config=embedding_config)

    user_emb = model.user_emb_table._cache
    item_emb = model.item_emb_table._cache

    assert (user_emb[0] != item_emb[0]).any()
    assert (user_emb[3] != item_emb[0]).any()
    assert (user_emb[0] != item_emb[3]).any()
