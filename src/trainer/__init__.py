from typing import Any, Dict

from .base_cf import CFTrainer
from .lightgcn_cls import GraphTrainer
from .nmf import NeuMFTrainer


def get_cf_trainer(
    num_users: int,
    num_items: int,
    config: Dict[str, Any],
) -> CFTrainer:
    model_config = config["model"]
    name = model_config["name"]
    if name == "nmf":
        return NeuMFTrainer(num_users, num_items, config)
    else:
        return GraphTrainer(num_users, num_items, config)
