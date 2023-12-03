from typing import Any, Dict, Optional

from loguru import logger

from src.dataset.avazu.avazu_fm import AvazuDataset
from src.dataset.avazu.avazu_on_ram import AvazuOnRam
from src.dataset.base import ICTRDataset
from src.dataset.criteo import get_dataset_cls


def get_ctr_dataset(
    dataloader_config: Dict[str, Any], train_info_to_val: Optional[Dict] = None
) -> ICTRDataset:
    if train_info_to_val is None:
        train_info_to_val = {}

    dataset_config: Dict = dataloader_config["dataset"]
    name = "criteo"
    if "name" in dataset_config:
        name = dataset_config.pop("name")

    if name == "criteo":
        dataset_cls = get_dataset_cls(dataloader_config)
    elif name == "avazu_on_ram" or name == "avazu":
        dataset_cls = AvazuOnRam
    elif name == "avazu_fm":
        dataset_cls = AvazuDataset

    logger.info(f"Datset cls: {dataset_cls}")
    return dataset_cls(**dataset_config, **train_info_to_val)
