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
    assert name in ["hccf", "lightgcn"]

    if name == "lightgcn":
        model = LightGCN(
            num_users,
            num_items,
            **model_config,
        )
    elif name == "hccf":
        model = HCCFModelCore(
            num_users,
            num_items,
            **model_config,
        )
    else:
        raise NotImplementedError()

    model_config["name"] = name

    return model
