from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.dataset.cf_graph_dataset import CFGraphDataset


class ICFTrainer(ABC):
    @abstractmethod
    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
    ):
        """
        Args:
            num_users: Number of users. This should be calculated from train dataset
            num_items: Number of items. This should be calculated from train dataset
            config: Config from .yaml file
        """
        ...

    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Args:
            dataloader: Train dataloader
                should be dataloader of CFGraphDataset

        Returns:
            the main loss is stored at {"loss"},
            while the other sub loss are stored in other value
        E.g.:
            {
                "loss": 0.2,
                "loss_a": 0.1,
                ...
            }
        """
        ...

    @abstractmethod
    def validate_epoch(
        self,
        train_dataset: CFGraphDataset,
        dataloader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Args:
            train_dataset: CFGraphDataset
                used to filter interacted items from the original list
            dataloader: validate dataloader
                should be dataloader of TestCFGraphDataset
            metrics supports from set: {"ndcg", "recall"}

        Returns: By default only calculate ndcg@20
        """

        ...

    @property
    def model(self):
        ...
