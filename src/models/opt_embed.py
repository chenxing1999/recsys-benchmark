import random
from collections import namedtuple
from functools import lru_cache, partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F


def get_mask(hidden_size: int):
    # matrix size  = hidden_size * hidden_size
    # Number of ones = \sum i with i from 1 to hidden_size + 1
    # then drop it

    values = torch.ones((hidden_size, hidden_size), dtype=torch.bool)
    matrix = torch.tril(values)

    return matrix


class OptEmbed(nn.Module):
    def __init__(self, num_item, hidden_size):
        super().__init__()
        self._num_item = num_item
        self._hidden_size = hidden_size
        self._weight = nn.Parameter(torch.empty((num_item, hidden_size)))
        nn.init.xavier_uniform_(self._weight)

        self._full_mask = get_mask(hidden_size)

        self._cur_weight = None

    def get_weight(self, mask_d: Optional[torch.Tensor] = None):
        device = self._weight.data.device
        self._full_mask = self._full_mask.to(device)

        if self.training:
            if mask_d is None:
                indices = torch.randint(
                    0, self._hidden_size, (self._num_item,), device=device
                )
                mask_d = F.embedding(indices, self._full_mask)

            self._cur_weight = self._weight * mask_d
        else:
            if mask_d is None:
                self._cur_weight = self._weight.data
            else:
                if isinstance(mask_d, (torch.IntTensor, torch.LongTensor)):
                    mask_d = mask_d.to(device)
                    mask_d = F.embedding(mask_d, self._full_mask)
                mask_d = mask_d.to(self._weight)
                self._cur_weight = self._weight * mask_d

        return self._cur_weight

    def forward(self, x, mask_d=None):
        if self._cur_weight is None:
            self.get_weight(mask_d)

        return F.embedding(x, self._cur_weight)


# Evo
Candidate = LightGCNCandidate = namedtuple("Candidate", ["item_mask", "user_mask"])


def _generate_lightgcn_candidate(num_user, num_item, hidden_size, target_sparsity=None):
    candidate = LightGCNCandidate(
        user_mask=_sampling_by_weight(target_sparsity, hidden_size, num_user),
        item_mask=_sampling_by_weight(target_sparsity, hidden_size, num_item),
    )

    cur_sparsity = _get_sparsity(candidate, hidden_size)

    while cur_sparsity < target_sparsity:
        candidate = LightGCNCandidate(
            user_mask=_sampling_by_weight(
                target_sparsity * 1.01, hidden_size, num_user
            ),
            item_mask=_sampling_by_weight(
                target_sparsity * 1.01, hidden_size, num_item
            ),
        )
        # Decrease alpha
        logger.debug(f"gen {cur_sparsity}")
    return candidate


def _validate_candidate(
    model,
    candidate: Candidate,
    val_loader,
    train_dataset,
):
    # this import here because it kind of weird for OptEmbed
    # to be depends on trainer

    from src.trainer.lightgcn import validate_epoch

    # hook
    model.item_emb_table.get_weight = partial(
        model.item_emb_table.get_weight,
        mask_d=candidate.item_mask,
    )
    model.user_emb_table.get_weight = partial(
        model.user_emb_table.get_weight,
        mask_d=candidate.user_mask,
    )
    # validate
    result = validate_epoch(train_dataset, val_loader, model)

    # unhook
    model.item_emb_table.get_weight = model.item_emb_table.get_weight.func
    model.user_emb_table.get_weight = model.user_emb_table.get_weight.func
    return result["ndcg"]


def _crossover(
    cur_top_candidate: List[Candidate],
    n_crossover: int,
    num_user,
    num_item,
    hidden_size,
    target_sparsity: Optional[float] = None,
) -> List[Candidate]:
    result = []
    for _ in range(n_crossover):
        while True:
            father, mother = random.choices(cur_top_candidate, k=2)

            # mix user
            father_user = father.user_mask
            mother_user = mother.user_mask
            son_user = torch.empty_like(father_user)
            father_choice_mask = torch.randint(2, size=(num_user,), dtype=bool)
            mother_choice_mask = torch.logical_not(father_choice_mask)
            son_user[father_choice_mask] = father_user[father_choice_mask]
            son_user[mother_choice_mask] = mother_user[mother_choice_mask]

            # mix item
            father_item = father.item_mask
            mother_item = mother.item_mask
            son_item = torch.empty_like(father_item)
            father_choice_mask = torch.randint(2, size=(num_item,), dtype=bool)
            mother_choice_mask = torch.logical_not(father_choice_mask)
            son_item[father_choice_mask] = father_item[father_choice_mask]
            son_item[mother_choice_mask] = mother_item[mother_choice_mask]

            candidate = Candidate(item_mask=son_item, user_mask=son_user)

            if target_sparsity is None:
                break

            cur_sparsity = _get_sparsity(candidate, hidden_size)
            if cur_sparsity > target_sparsity:
                break
            logger.debug(f"crossover... {cur_sparsity}")

        logger.debug("crossovered", cur_sparsity)
        result.append(candidate)

    return result


def _mutate(
    cur_top_candidate: List[Candidate],
    n_mutate: int,
    p_mutate: float,
    num_user: int,
    num_item: int,
    hidden_size: int,
    target_sparsity: Optional[float] = None,
) -> List[Candidate]:
    result = []

    max_hidden_size_budget = hidden_size
    for _ in range(n_mutate):
        while True:
            parent = random.choice(cur_top_candidate)

            son_item = parent.item_mask.clone()
            mask = torch.rand(son_item.shape[0]) < p_mutate
            num_mutated = mask.sum()
            son_item[mask] = _sampling_by_weight(
                target_sparsity, hidden_size, num_mutated
            )
            son_user = parent.user_mask.clone()
            mask = torch.rand(son_user.shape[0]) < p_mutate
            num_mutated = mask.sum()
            son_user[mask] = _sampling_by_weight(
                target_sparsity, hidden_size, num_mutated
            )

            candidate = Candidate(item_mask=son_item, user_mask=son_user)

            if target_sparsity is None:
                break

            cur_sparsity = _get_sparsity(candidate, hidden_size)
            logger.debug(f"mutating... {cur_sparsity}")
            if cur_sparsity > target_sparsity:
                break

            max_hidden_size_budget -= 1

        logger.debug("mutated", cur_sparsity)
        result.append(candidate)

    return result


def _get_sparsity(candidate: Candidate, hidden_size):
    # mask count from 0
    n_elements = (candidate.user_mask + 1).sum() + (candidate.item_mask + 1).sum()

    num_item = len(candidate.item_mask)
    num_user = len(candidate.user_mask)
    n_max_elements = (num_item + num_user) * hidden_size

    cur_sparsity = 1 - n_elements / n_max_elements
    return cur_sparsity


def evol_search_lightgcn(
    model,
    n_generations: int,
    population: int,
    n_crossover: int,
    n_mutate: int,
    p_mutate: float,
    k: int,
    val_dataloader,
    train_dataset,
    target_sparsity=None,
) -> Tuple[torch.LongTensor, torch.LongTensor, float]:
    """Evolutionary search for LightGCN with OptEmbed

    Args:
        model: LightGCN model with OptEmbed
        n_generations: number of generations to run
        population: Starting population

        n_crossover: number of crossovers per generations
        n_mutate: number of mutate per generations
        p_mutate: Probability to mutate
        k: How many best candidates are kept per generations

        val_dataloader: DataLoader for validation dataset
        train_dataset: Train dataset to get information such
            as num_item, num_users, ...

        target_sparsity: Maximum sparsity accepted

    Returns:
        best_item_mask: (torch.LongTensor, shape (num_items,))
            best_item_mask[i] = how many dimension assigned for item i

        best_user_mask: (torch.LongTensor, shape (num_users,))
        best_ndcg (float)
    """

    cur_top_values = None
    cur_top_candidate = []

    num_items = train_dataset.num_items
    num_users = train_dataset.num_users

    assert isinstance(model.item_emb_table, OptEmbed)
    assert isinstance(model.user_emb_table, OptEmbed)
    hidden_size = model.item_emb_table._hidden_size

    candidates = [
        _generate_lightgcn_candidate(num_users, num_items, hidden_size, target_sparsity)
        for _ in range(population)
    ]

    for gen in range(n_generations):
        logger.debug(f"start {gen=}")
        metrics = torch.tensor(
            [
                _validate_candidate(model, candidate, val_dataloader, train_dataset)
                for candidate in candidates
            ]
        )
        if cur_top_values is not None:
            cur_top_values = torch.cat((cur_top_values, metrics))
        else:
            cur_top_values = metrics
        cur_top_candidate.extend(candidates)

        result = torch.topk(cur_top_values, k)

        cur_top_candidate = [cur_top_candidate[idx] for idx in result.indices]
        cur_top_values = result.values

        logger.debug(f"cur best {cur_top_candidate[0]} - {cur_top_values[0]}")

        if gen != n_generations - 1:
            logger.debug(f"mutate and crossover {gen=}")
            candidates = []
            crossovers = _crossover(
                cur_top_candidate,
                n_crossover,
                num_users,
                num_items,
                hidden_size,
                target_sparsity,
            )
            candidates.extend(crossovers)

            mutates = _mutate(
                cur_top_candidate,
                n_mutate,
                p_mutate,
                num_users,
                num_items,
                hidden_size,
                target_sparsity,
            )
            candidates.extend(mutates)

    top_candidate = cur_top_candidate[0]
    item_mask = top_candidate.item_mask
    user_mask = top_candidate.user_mask

    return item_mask, user_mask, cur_top_values[0]


@lru_cache(1)
def _find_alpha(target_sparsity, step=0.99):
    if target_sparsity == 0.7:
        return 0.989
    elif target_sparsity == 0.8:
        return 0.9801
    elif target_sparsity == 0.5:
        return 1

    alpha = 1
    # large number for sampling logic
    h = 256
    while True:
        f = np.power(alpha, np.arange(1, h + 1) * (-1) + h)
        p = f / f.sum()

        # print(p)
        E = (np.arange(1, h + 1) * p / h).sum()
        if E > target_sparsity:
            return alpha
        alpha = alpha * 0.99


def _generate_weight(alpha, hidden_size):
    f = np.power(alpha, np.arange(1, hidden_size + 1) * (-1) + hidden_size)
    p = f / f.sum()
    return p


def _sampling_by_weight(target_sparsity, hidden_size, num_item):
    if target_sparsity is None:
        return torch.randint(0, hidden_size, (num_item,))

    alpha = _find_alpha(target_sparsity)
    weight = _generate_weight(alpha, hidden_size)
    sampler = torch.utils.data.WeightedRandomSampler(weight, num_item)
    return torch.tensor(list(sampler))


# Retrain
