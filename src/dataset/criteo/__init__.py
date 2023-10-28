#!
from typing import Dict, Type, Union

from .criteo import CriteoDataset
from .criteo_iter import CriteoIterDataset
from .criteo_torchfm import CriteoDataset as CriteoFMDataset

CriteoDatasetType = Union[
    Type[CriteoDataset], Type[CriteoIterDataset], Type[CriteoFMDataset]
]


def get_dataset_cls(loader_config: Dict) -> CriteoDatasetType:
    num_workers = loader_config.get("num_workers", 0)
    shuffle = loader_config.get("shuffle", False)

    if "train_test_info" in loader_config["dataset"]:
        return CriteoFMDataset
    if num_workers == 0 and not shuffle:
        return CriteoIterDataset
    else:
        return CriteoDataset
