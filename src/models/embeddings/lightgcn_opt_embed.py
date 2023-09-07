import random
from collections import namedtuple
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from .base import IEmbedding
from .optembed_utils import _MaskEmbeddingModule, _sampling_by_weight, get_mask


class IOptEmbed(IEmbedding):
    def get_weight(self, mask_d: Optional[torch.Tensor] = None):
        ...

    def forward(self, x, mask_d=None):
        ...

    def get_l_s(self):
        return 0


class OptEmbed(IOptEmbed):
    """Opt Embed wrapper for paper
    https://arxiv.org/abs/2208.04482

    Note: In this implementation, I use a cache mechanism.

    If you call `get_weight` or `forward`:
        the cache (weight with masked value)
            will be created

    if you call `backward`:
        the cache will be deleted
        This is because the change of masked value
    """

    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size: int,
        mode: Optional[str] = None,
        t_init: Optional[float] = 0,
        mode_threshold_e="field",
        mode_threshold_d="field",
        norm=1,
        target_sparsity: Optional[float] = None,
    ):
        """
        Args:
            field_dims: List of dimension size
            hidden_size: size of vector output
            mode in ["sum", "mean", "max", None]. Reduction method
                see torch.nn.EmbeddingBag for more information

            t_init: Initialize value of t vector
                if None, remove mask embedding

            mode_threshold_e: Literal["field","feature"]
                if mode="field", value t of feature in the same field will be the same

            mode_threshold_d: Literal["field","feature"]
                if mode="field", dimension of feature in the same field will be the same

            norm: norm = 1 --> L1, norm = 2 --> L2

            target_sparsity:
                 determined sampling logic.
                 If target_sparsity is None, use original sampling logic
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in ["sum", "mean", "max", None]
        assert mode_threshold_e in ["field", "feature"]
        assert mode_threshold_d in ["field", "feature"]

        self._field_dims = torch.tensor(field_dims, dtype=torch.int64)
        self._num_item = self._field_dims.sum()
        self._num_field = len(field_dims)

        self._hidden_size = hidden_size

        self._weight = nn.Parameter(torch.empty((self._num_item, hidden_size)))
        self._cur_weight = None
        nn.init.xavier_uniform_(self._weight)

        # When backward -> weight update
        # value of cur weight should be changing
        self._handle = self.register_full_backward_hook(_delete_cache)
        self._mode = mode

        # Mask E related logic
        self._t_init = t_init
        if t_init is None:
            self._mask_e_module = nn.Identity()
        else:
            self._mask_e_module = _MaskEmbeddingModule(
                self._field_dims,
                t_init,
                mode_threshold_e,
                norm,
            )

        # Mask D
        # use buffer so that when move model to device,
        # _full_mask_d will get assign correct device
        _full_mask_d = get_mask(hidden_size)
        self.register_buffer("_full_mask_d", _full_mask_d)

        self._target_sparsity = target_sparsity
        self._naive = False
        self._mode_d = mode_threshold_d

    def get_l_s(self):
        if self._t_init is None:
            return 0
        return torch.exp(-self._mask_e_module._t_param).sum()

    def get_weight(self, mask_d: Optional[torch.Tensor] = None):
        device = self._weight.data.device

        # Apply mask_e
        emb = self._mask_e_module(self._weight)

        if not self.training and mask_d is None:
            self._cur_weight = emb
            return self._cur_weight

        if self.training:
            if mask_d is None:
                # Sampling mask d
                if self._mode_d == "feature":
                    indices = _sampling_by_weight(
                        self._target_sparsity,
                        self._hidden_size,
                        self._num_item,
                        self._naive,
                        device,
                    )
                elif self._mode_d == "field":
                    # Only support uniform for now
                    indices = torch.randint(0, self._hidden_size, (self._num_field,))
                    indices = torch.repeat_interleave(
                        indices,
                        self._field_dims,
                        dim=0,
                        output_size=self._num_item,
                    ).to(device)

                mask_d = F.embedding(indices, self._full_mask_d)
            elif isinstance(mask_d, torch.LongTensor):
                mask_d = F.embedding(mask_d, self._full_mask_d)

            self._cur_weight = emb * mask_d
        else:
            if isinstance(mask_d, (torch.IntTensor, torch.LongTensor)):
                mask_d = mask_d.to(device)
                field_dims = self._field_dims.to(device)
                if self._mode_d == "field":
                    mask_d = torch.repeat_interleave(
                        mask_d,
                        field_dims,
                        dim=0,
                        output_size=self._num_item,
                    )

                mask_d = F.embedding(mask_d, self._full_mask_d)
            mask_d = mask_d.to(self._weight)
            self._cur_weight = emb * mask_d

        return self._cur_weight

    def forward(self, x, mask_d=None):
        """
        Args:
            x: (torch.LongTensor B x Num field) indices of feature after offsets
            mask_d:
                if ([torch.LongTensor], shape: Num feature)
                    mask_d[i] + 1 = dimension used for feature i
                if ([torch.BoolTensor], shape: Num feature x Dimension)
                    directly used as a mask
        """
        # Creating self._cur_weight
        if self._cur_weight is None:
            self.get_weight(mask_d)

        mode = self._mode
        if mode is None:
            return F.embedding(x, self._cur_weight)
        else:
            return F.embedding_bag(x, self._cur_weight, mode=mode)

    def get_sparsity(self, get_n_params=False):
        emb = self._mask_e_module(self._weight)
        nnz = torch.nonzero(emb).size(0)
        nnz = int(nnz)
        sparsity = 1 - nnz / (emb.shape[0] * emb.shape[1])
        if get_n_params:
            return sparsity, nnz
        return sparsity


# Evo
Candidate = LightGCNCandidate = namedtuple("Candidate", ["item_mask", "user_mask"])


def _generate_lightgcn_candidate(
    num_user, num_item, hidden_size, target_sparsity=None, naive=False
):
    candidate = LightGCNCandidate(
        user_mask=_sampling_by_weight(
            target_sparsity,
            hidden_size,
            num_user,
            naive,
        ),
        item_mask=_sampling_by_weight(
            target_sparsity,
            hidden_size,
            num_item,
            naive,
        ),
    )

    if target_sparsity is None:
        return candidate
    cur_sparsity = _get_sparsity(candidate, hidden_size)
    step = 1.05
    while cur_sparsity < target_sparsity:
        candidate = LightGCNCandidate(
            user_mask=_sampling_by_weight(
                target_sparsity * step, hidden_size, num_user
            ),
            item_mask=_sampling_by_weight(
                target_sparsity * step, hidden_size, num_item
            ),
        )
        cur_sparsity = _get_sparsity(candidate, hidden_size)
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
    naive=False,
) -> List[Candidate]:
    result = []

    max_hidden_size_budget = hidden_size
    for _ in range(n_mutate):
        while True:
            parent = random.choice(cur_top_candidate)

            son_item = parent.item_mask.clone()
            mask = torch.rand(son_item.shape[0]) < p_mutate
            num_mutated = mask.sum().item()
            son_item[mask] = _sampling_by_weight(
                target_sparsity,
                hidden_size,
                num_mutated,
                naive,
            )
            son_user = parent.user_mask.clone()
            mask = torch.rand(son_user.shape[0]) < p_mutate
            num_mutated = mask.sum().item()
            son_user[mask] = _sampling_by_weight(
                target_sparsity,
                hidden_size,
                num_mutated,
                naive,
            )

            candidate = Candidate(item_mask=son_item, user_mask=son_user)

            if target_sparsity is None:
                break

            cur_sparsity = _get_sparsity(candidate, hidden_size)
            if cur_sparsity > target_sparsity:
                break

            max_hidden_size_budget -= 1

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
    naive=False,
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
        naive: Generate with target sparsity with linearly
            reduce max hidden size

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

    assert isinstance(model.item_emb_table, IOptEmbed)
    assert isinstance(model.user_emb_table, IOptEmbed)
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

        cur_best_sparsity = _get_sparsity(cur_top_candidate[0], hidden_size)
        logger.debug(
            f"cur best {cur_top_candidate[0]}"
            f"- ndcg: {cur_top_values[0]:.4f}"
            f"- sparsity: {cur_best_sparsity:.4f}"
        )

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


def _delete_cache(module: OptEmbed, grad_input, grad_output):
    module._cur_weight = None


# Retrain
