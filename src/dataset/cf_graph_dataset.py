import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ..graph_utils import calculate_sparse_graph_adj_norm


def load_graph_dataset(path: str) -> Tuple[dict, list, int, int]:
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
    user_interact_pair = []
    with open(path) as fin:
        for line in fin.readlines():
            info = line.strip().split()
            user_id = int(info[0])
            interacted_items = [int(item) for item in info[1:]]
            if len(interacted_items) == 0:
                print(
                    f"Not found item for user {user_id=} in {line=}.",
                    "Remove from dataset",
                )
                continue

            graph[user_id] = interacted_items
            users.append(user_id)
            num_item = max(*graph[user_id], num_item)
            num_interactions += len(interacted_items)
            user_interact_pair += [user_id] * len(interacted_items)

    # num_item is currently max item id
    # item_id count from 0 --> To get num_item, need to plus 1
    return graph, users, num_item + 1, user_interact_pair


class CFGraphDataset(Dataset):
    def __init__(self, path: str):
        """
        Args:
            path: Path to dataset txt file
                the file contains multiple line with each line contains
                    <user_id> <item_id1> <item_id2> ...
        """

        self._path = path
        graph, users, num_item, user_interact_pair = load_graph_dataset(path)

        self._users = users
        self._users_interact_pair = user_interact_pair
        self._graph = graph
        self._num_item = num_item
        self._norm_adj = calculate_sparse_graph_adj_norm(
            self._graph, self._num_item, len(self._users)
        )
        self.dataset_length = user_interact_pair // len(self._users)

    def __len__(self):
        # return len(self._users)
        # return len(self._users_interact_pair)
        return self.dataset_length

    def __getitem__(self, idx) -> Tuple[int, int, int]:
        user_idx = random.randint(0, self.num_users - 1)
        if not self._graph[user_idx]:
            raise ValueError("Exists user with no item interaction")

        pos_item_idx: int = random.choice(self._graph[user_idx])

        neg_item_idx = pos_item_idx
        while neg_item_idx in self._graph[user_idx]:
            neg_item_idx = random.randint(0, self.num_items - 1)
        return user_idx, pos_item_idx, neg_item_idx

    def describe(self):
        # Minor analyze for dataset
        print("Num user:", len(self._users), "- Num item:", self._num_item)

        stats = []
        for items in self._graph.values():
            stats.append(len(items))

        print("Min degree", min(stats))
        print("Max degree", max(stats))

    @property
    def num_users(self):
        return len(self._users)

    @property
    def num_items(self):
        return self._num_item

    def get_graph(self) -> Dict[int, List[int]]:
        return self._graph


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
        graph, users, num_item, num_interactions = load_graph_dataset(path)

        self._users = users
        self._graph = graph
        self._num_item = num_item

    def __len__(self):
        return len(self._users)

    def __getitem__(self, idx) -> Tuple[int, List[int]]:
        """
        Return:
            user_id
            all_item_pos
        """
        user_idx = self._users[idx]
        if not self._graph[user_idx]:
            raise ValueError("Exists user with no item interaction")

        return user_idx, self._graph[user_idx]

    @staticmethod
    def collate_fn(batch, pad_idx=-1):
        new_users = []
        new_user_items = []
        for user, items in batch:
            new_users.append(user)
            new_user_items.append(items)

        return torch.tensor(new_users), new_user_items
