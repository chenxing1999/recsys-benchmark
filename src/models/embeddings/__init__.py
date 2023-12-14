import copy
from typing import Dict, List, Optional, Union

from loguru import logger

from .base import IEmbedding, VanillaEmbedding
from .cerp_embedding import CerpEmbedding, RetrainCerpEmbedding
from .deepfm_opt_embed import OptEmbed as DeepFMOptEmbed
from .deepfm_opt_embed import RetrainOptEmbed as DeepFMRetrainOptEmbed
from .dh_embedding import DHEmbedding
from .lightgcn_opt_embed import OptEmbed, RetrainOptEmbed
from .pep_embedding import PepEmbeeding, RetrainPepEmbedding
from .qr_embedding import QRHashingEmbedding
from .tensortrain_embeddings import TTEmbedding, TTRecTorch  # type: ignore

NAME_TO_CLS = {
    "vanilla": VanillaEmbedding,
    "qr": QRHashingEmbedding,
    "dhe": DHEmbedding,
    "pep": PepEmbeeding,
    "pep_retrain": RetrainPepEmbedding,
    "optembed_d": OptEmbed,  # will only use mask D
    "optembed_d_retrain": RetrainOptEmbed,  # will only use mask D
    "optembed": OptEmbed,
    "optembed_retrain": RetrainOptEmbed,
    "deepfm_optembed": DeepFMOptEmbed,
    "deepfm_optembed_retrain": DeepFMRetrainOptEmbed,
    "tt_emb": TTEmbedding,
    "tt_emb_torch": TTRecTorch,
    "cerp": CerpEmbedding,
    "cerp_retrain": RetrainCerpEmbedding,
}


def get_embedding(
    embedding_config: Dict,
    field_dims: Union[int, List[int]],
    hidden_size: int,
    mode: Optional[str] = None,
    field_name: str = "",
) -> IEmbedding:
    assert mode in [None, "sum", "mean", "max"], "Unsupported mode"
    name = embedding_config["name"]
    embedding_config = copy.deepcopy(embedding_config)
    embedding_config.pop("name")

    name_to_cls = NAME_TO_CLS

    if name == "vanilla":
        emb = VanillaEmbedding(
            field_dims,
            hidden_size,
            mode=mode,
            **embedding_config,
        )
    elif name not in name_to_cls:
        raise NotImplementedError(f"{name} not found in mapping from name to class")
    else:
        if name.startswith("pep") or name.startswith("cerp"):
            embedding_config["field_name"] = field_name
        if name == "optembed_d" or name == "optembed_d_retrain":
            embedding_config["t_init"] = None
            logger.debug("OptEmbed Mask E is disabled before creating")

        cls = name_to_cls[name]
        emb = cls(field_dims, hidden_size, mode=mode, **embedding_config)

    embedding_config["name"] = name
    return emb
