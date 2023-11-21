from typing import Any, Dict, Optional

from src.dataset.base import ICTRDataset


def get_ctr_dataset(
    dataloader_config: Dict[str, Any], train_info_to_val: Optional[Dict] = None
) -> ICTRDataset:
    raise NotImplementedError()
