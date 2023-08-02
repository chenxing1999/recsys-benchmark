from typing import Dict

from .base import IGraphBaseCore
from .hccf import HCCFModelCore
from .lightgcn import LightGCN


def get_graph_model(
    num_users: int,
    num_items: int,
    model_config: Dict,
) -> IGraphBaseCore:
    """Wrapper to get model from config"""

    # pop name for ** trick
    name = model_config.pop("name")

    name_to_cls = {
        "lightgcn": LightGCN,
        "hccf": HCCFModelCore,
    }
    assert name in name_to_cls
    cls = name_to_cls[name]
    model = cls(num_users, num_items, **model_config)

    model_config["name"] = name

    return model
