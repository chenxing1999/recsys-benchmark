import copy
from typing import Dict

from src.models.opt_embed import OptEmbed

from .base import IEmbedding, VanillaEmbedding
from .dh_embedding import DHEmbedding
from .pep_embedding import PepEmbeeding, RetrainPepEmbedding
from .qr_embedding import QRHashingEmbedding


def get_embedding(
    embedding_config: Dict,
    num_item: int,
    hidden_size: int,
    field_name: str = "",
) -> IEmbedding:
    name = embedding_config["name"]
    embedding_config = copy.deepcopy(embedding_config)
    embedding_config.pop("name")

    name_to_cls = {
        "vanilla": VanillaEmbedding,
        "qr": QRHashingEmbedding,
        "dhe": DHEmbedding,
        "pep": PepEmbeeding,
        "pep_retrain": RetrainPepEmbedding,
    }
    if name == "vanilla":
        emb = VanillaEmbedding(
            num_item,
            hidden_size,
            **embedding_config,
        )
    elif name not in name_to_cls:
        raise NotImplementedError()
    elif name == "optembed":
        emb = OptEmbed(num_item, hidden_size)
    else:
        if name.startswith("pep"):
            embedding_config["field_name"] = field_name
        cls = name_to_cls[name]
        emb = cls(num_item, hidden_size, **embedding_config)

    embedding_config["name"] = name
    return emb
