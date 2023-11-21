from typing import Any, Dict, Optional

from src.dataset.base import ICTRDataset
from src.dataset.criteo import get_dataset_cls


def get_ctr_dataset(
    dataloader_config: Dict[str, Any], train_info_to_val: Optional[Dict] = None
) -> ICTRDataset:
    if train_info_to_val is None:
        train_info_to_val = {}

    dataset_cls = get_dataset_cls(dataloader_config)
    return dataset_cls(**dataloader_config, **train_info_to_val)
