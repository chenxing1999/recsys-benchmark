from typing import Dict, Type

import torch

from src.models.lightgcn import LightGCN, SingleLightGCN

from .base import IGraphBaseCore
from .hccf import HCCFModelCore


def get_graph_model(
    num_users: int,
    num_items: int,
    model_config: Dict,
) -> IGraphBaseCore:
    """Wrapper to get model from config"""

    # pop name for ** trick
    name = model_config.pop("name")

    name_to_cls: Dict[str, Type[IGraphBaseCore]] = {
        "lightgcn": LightGCN,
        "single-lightgcn": SingleLightGCN,
        "hccf": HCCFModelCore,
    }
    assert name in name_to_cls
    cls = name_to_cls[name]
    model = cls(num_users, num_items, **model_config)

    model_config["name"] = name

    return model


def load_graph_model(checkpoint_path: str) -> IGraphBaseCore:
    """Load checkpoint to correct model"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    model_config = checkpoint["model_config"]

    model = get_graph_model(num_users, num_items, model_config)
    model.load_state_dict(checkpoint["state_dict"])
    return model
