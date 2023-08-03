import random
from typing import Dict, List, Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset

from ..graph_utils import calculate_sparse_graph_adj_norm, get_adj


def load_graph_dataset(path: str) -> Tuple[dict, list, int, list]:
    """

    Returns:
        graph: (dict[int, list[int]]) Graph map from user index to
            list of items interacted
        users: (list[int])
        num_item: (int)
    """
    graph = {}
    users = []
    num_item = 0
    num_interactions = 0
    user_item_pairs = []
    with open(path) as fin:
        for line in fin.readlines():
            info = line.strip().split()
            user_id = int(info[0])
            interacted_items = [int(item) for item in info[1:]]
            if len(interacted_items) == 0:
                logger.warning(
                    f"Not found item for user {user_id=} in {line=}.",
                    "Remove from dataset",
                )
                continue

            graph[user_id] = interacted_items
            users.append(user_id)
            num_item = max(*graph[user_id], num_item)
            num_interactions += len(interacted_items)
            user_item_pairs.extend((user_id, item) for item in interacted_items)

    # num_item is currently max item id
    # item_id count from 0 --> To get num_item, need to plus 1
    return graph, users, num_item + 1, user_item_pairs


class CFGraphDataset(Dataset):
    """LightGCN Graph CF Structure"""

    def __init__(self, path: str, adj_style: str, sampling_method="uniform"):
        """
        Args:
            path: Path to dataset txt file
                the file contains multiple line with each line contains
                    <user_id> <item_id1> <item_id2> ...
            adj_style: lightgcn or hccf
        """
        assert adj_style in [
            "lightgcn",
            "hccf",
        ], f"{adj_style=}, only accepts ['lightgcn', 'hccf']"

        assert sampling_method in ["uniform", "popularity"]

        self._path = path
        graph, users, num_item, user_item_pairs = load_graph_dataset(path)
        num_interactions = len(user_item_pairs)

        self._users = users
        self._num_interactions = num_interactions
        self._graph = graph

        self._user_item_pairs = user_item_pairs
        self._num_item = num_item
        self.sampling_method = sampling_method

        if adj_style == "lightgcn":
            self._norm_adj = calculate_sparse_graph_adj_norm(
                self._graph, self.num_items, self.num_users
            )
        elif adj_style == "hccf":
            self._norm_adj = get_adj(
                graph,
                self.num_items,
                self.num_users,
                normalize=True,
            )

        else:
            raise ValueError(f"{adj_style=} is not supported")
        self.per_user_num = num_interactions // self.num_users
        self.dataset_length = self.num_users * self.per_user_num

    def __len__(self):
        sampling_method = self.sampling_method
        if sampling_method == "uniform":
            return self.dataset_length
        elif sampling_method == "popularity":
            return len(self._user_item_pairs)

    def __getitem__(self, idx):
        sampling_method = self.sampling_method
        if self.sampling_method == "uniform":
            return self._get_uniform(idx)
        elif sampling_method == "popularity":
            return self._get_popularity(idx)

    def _get_popularity(self, idx) -> Tuple[int, int, int]:
        user_idx, pos_item_idx = self._user_item_pairs[idx]
        neg_item_idx = pos_item_idx

        while neg_item_idx in self._graph[user_idx]:
            neg_item_idx = random.randint(0, self.num_items - 1)
        return user_idx, pos_item_idx, neg_item_idx

    def _get_uniform(self, idx) -> Tuple[int, int, int]:
        user_idx = idx // self.per_user_num
        if not self._graph[user_idx]:
            raise ValueError("Exists user with no item interaction")

        pos_item_idx: int = random.choice(self._graph[user_idx])

        neg_item_idx = pos_item_idx
        while neg_item_idx in self._graph[user_idx]:
            neg_item_idx = random.randint(0, self.num_items - 1)
        return user_idx, pos_item_idx, neg_item_idx

    def describe(self):
        # Minor analyze for dataset
        msg = f"Num user: {self.num_users} - Num item: {self.num_items}"
        logger.info(msg)
        logger.info(f"Num interactions: {self._num_interactions}")
        logger.info(
            f"Sparsity: {self._num_interactions / (self.num_users * self.num_items)}"
        )

        stats = []
        for items in self._graph.values():
            stats.append(len(items))

        logger.info(f"Min degree - {min(stats)}")
        logger.info(f"Max degree - {max(stats)}")

    @property
    def num_users(self):
        return len(self._users)

    @property
    def num_items(self):
        return self._num_item

    def get_graph(self) -> Dict[int, List[int]]:
        return self._graph

    def get_norm_adj(self):
        return self._norm_adj


class TestCFGraphDataset(Dataset):
    def __init__(self, path: str):
        """
        Args:
            path: Path to dataset txt file
                the file contains multiple line with each line contains
                    <user_id> <item_id1> <item_id2> ...
        """
        self._path = path
        self._path = path
        graph, users, num_item, _ = load_graph_dataset(path)

        self._users = users
        self._graph = graph
        self._num_item = num_item
        self._compute_set()

    def __len__(self):
        return len(self._users)

    def __getitem__(self, idx) -> Tuple[int, List[int]]:
        """
        Return:
            user_id
            all_item_pos
            set_all_item_pos
        """
        user_idx = self._users[idx]
        if not self._graph[user_idx]:
            raise ValueError("Exists user with no item interaction")

        return user_idx, self._idx_to_set[user_idx]

    @staticmethod
    def collate_fn(batch, pad_idx=-1):
        new_users = []
        new_user_items = []
        for user, items in batch:
            new_users.append(user)
            new_user_items.append(items)

        return torch.tensor(new_users), new_user_items

    def _compute_set(self):
        self._idx_to_set = {}
        for idx, v in self._graph.items():
            self._idx_to_set[idx] = set(v)
