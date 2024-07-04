from typing import Any, Dict, Optional, Type

from loguru import logger

from src.dataset.avazu.avazu_fm import AvazuDataset
from src.dataset.avazu.avazu_on_ram import AvazuOnRam
from src.dataset.base import ICTRDataset
from src.dataset.criteo import get_dataset_cls
from src.dataset.kdd.kdd_dataset import KddDataset


def get_ctr_dataset(
    dataloader_config: Dict[str, Any], train_info_to_val: Optional[Dict] = None
) -> ICTRDataset:
    if train_info_to_val is None:
        train_info_to_val = {}

    dataset_config: Dict = dataloader_config["dataset"]
    name = "criteo"
    if "name" in dataset_config:
        name = dataset_config.pop("name")

    dataset_cls: Type[ICTRDataset]
    if name == "criteo":
        dataset_cls = get_dataset_cls(dataloader_config)
    elif name == "avazu_on_ram" or name == "avazu":
        dataset_cls = AvazuOnRam
    elif name == "avazu_fm":
        dataset_cls = AvazuDataset
    elif name == "kdd":
        dataset_cls = KddDataset

    logger.info(f"Datset cls: {dataset_cls}")
    return dataset_cls(**dataset_config, **train_info_to_val)
