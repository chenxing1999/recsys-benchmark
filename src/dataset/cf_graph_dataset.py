import random

from torch.utils.data import Dataset

from ..graph_utils import calculate_sparse_graph_adj_norm


class CF_Graph_Dataset(Dataset):
    def __init__(self, path: str):
        """
        Args:
            path: Path to dataset txt file
                the file contains multiple line with each line contains
                    <user_id> <item_id1> <item_id2> ...
        """
        self._path = path
        self._users = []

        # Graph map from user index to list of items interacted
        self._graph = {}
        num_item = 0
        with open(path) as fin:
            for line in fin.readlines():
                info = line.strip().split()
                user_id = int(info[0])
                self._graph[user_id] = [int(item) for item in info[1:]]
                self._users.append(user_id)
                num_item = max(*self._graph[user_id], num_item)

        # num_item is currently max item id
        # item_id count from 0 --> To get num_item, need to plus 1
        self._num_item = num_item + 1
        self._norm_adj = calculate_sparse_graph_adj_norm(
            self._graph, self._num_item, len(self._users)
        )

    def __len__(self):
        return len(self._users)

    def __getitem__(self, idx):
        user_idx = self._users[idx]
        if not self._graph[user_idx]:
            raise ValueError("Exists user with no item interaction")

        pos_item_idx = random.choice(self._graph[user_idx])

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
