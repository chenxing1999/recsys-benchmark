"""Original implementation for DeepFM OptEmbedding"""
import random
from collections import namedtuple
from functools import lru_cache
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

    def get_sparsity(self, get_n_params=False):
        """
        Args:
            get_n_params

        Returns: Tuple[float, int] or float
            if get_n_params,
                return sparsity, nnz
            else
                return sparsity
        """
        ...


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

        Note: main difference of DeepFM OptEmbed and LightGCN OptEmbed
            - DeepFM will generate mask dimension with shape Batch x Num field
            - LightGCN will only generate mask dimension for all embedding
                If apply LightGCN forward, the mask dimension will be Num field.
        """
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert mode in ["sum", "mean", "max", None]
        assert mode_threshold_e in ["field", "feature"]
        assert mode_threshold_d in ["field", "feature"]

        self._field_dims = torch.tensor(field_dims, dtype=torch.int64)

        start_field_idx = torch.concat([torch.tensor([0], dtype=int), self._field_dims])

        # self._offsets: 0, 2, 5
        self._offsets = torch.cumsum(start_field_idx, 0)[:-1]
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
        device = self._weight.data.device
        b = x.shape[0]
        mode = self._mode

        if self.training:
            x = F.embedding(x, self._weight)
            emb = self._mask_e_module(x)
            mask_d = torch.randint(
                0, self._hidden_size, size=(b, self._num_field), device=device
            )
            mask_d = F.embedding(mask_d, self._full_mask_d)
            emb = mask_d * emb
            if mode is None:
                return emb
            elif mode == "sum":
                return emb.sum(1)
            elif mode == "max":
                return emb.max(1)
            elif mode == "mean":
                return emb.mean(1)

        # Evaluation forward logic
        if self._cur_weight is None:
            self.get_weight(mask_d)

        if self._mode is None:
            return F.embedding(x, self._cur_weight)
        else:
            return F.embedding_bag(x, self._cur_weight, mode=self._mode)

    def get_sparsity(self, get_n_params=False):
        emb = self._mask_e_module(self._weight)
        nnz = torch.nonzero(emb).size(0)
        nnz = int(nnz)
        sparsity = 1 - nnz / (emb.shape[0] * emb.shape[1])
        if get_n_params:
            return sparsity, nnz
        return sparsity

    def get_num_params(self):
        emb = self._mask_e_module(self._weight)
        nnz = torch.nonzero(emb).size(0)
        nnz = int(nnz)
        return nnz

    def get_mask_e(self):
        if isinstance(self._mask_e_module, nn.Identity):
            return torch.ones(self._num_item, dtype=int)

        # apply mask e to weight
        emb = self._mask_e_module(self._weight)

        # mask_e[i] = 1 if still use feature i, else 0
        # mask_e shape=num_feature
        mask_e = (emb.norm(1, 1) > 0).to(int)
        return mask_e.cpu()

    @lru_cache(1)
    def get_submask(self) -> torch.LongTensor:
        """Return submask
            submask[i] is num feature used that mask_d[i] represent
        so num_element with a mask d = (submask * mask_d).sum()

        submask.shape = num field if mode_d is field
                      = num feature if mode_d is feature
        """
        if isinstance(self._mask_e_module, nn.Identity):
            if self._mode_d == "field":
                return self._field_dims
            else:
                return torch.ones(self._num_item, dtype=int)

        # apply mask e to weight
        emb = self._mask_e_module(self._weight)

        # mask_e[i] = 1 if still use feature i, else 0
        # mask_e shape=num_feature
        mask_e = (emb.norm(1, 1) > 0).to(int)
        if self._mode_d == "feature":
            return mask_e.cpu()

        # num_e[i] = num element used until index i (shape=num_feature + 1)
        # (num_e[0] = 0 and num_e[-1]=num feature left after apply mask e)
        num_e = mask_e.cumsum(0).cpu()
        num_e = torch.cat([torch.tensor([0], dtype=int), num_e])

        start_idx = torch.cat([torch.tensor([0], dtype=int), self._field_dims])
        offsets = start_idx.cumsum(0)

        return num_e[offsets[1:]] - num_e[offsets[:-1]]


# Evo
# `extra` used to calculate sparsity
Candidate = DeepFMCandidate = namedtuple("Candidate", ["save_mask", "extra"])


def _generate_candidate(
    emb: OptEmbed,
    target_sparsity=None,
    d_target_sparsity=None,
    method=1,
):
    """Generate Candidate for DeepFM mask D

    Args:
        emb
        target_sparsity: Target sparsity to get
            only sample candidate when the sparsity lower than target

        method:
            0: Uniform (original in paper)
            1: exponential
            2: linear

    """
    if d_target_sparsity is None and target_sparsity is not None:
        d_target_sparsity = target_sparsity

    hidden_size = emb._hidden_size
    mode_threshold_d = emb._mode_d

    if mode_threshold_d == "field":
        size = emb._num_field
    elif mode_threshold_d == "feature":
        size = emb._num_item
    else:
        raise NotImplementedError()

    with torch.no_grad():
        sub_mask = emb.get_submask()

    n_max = emb._num_item * hidden_size

    extra = (sub_mask, n_max)
    mask = _sampling_by_weight(
        d_target_sparsity,
        hidden_size,
        size,
        method,
    )

    # Get true field dims per fields
    candidate = DeepFMCandidate(
        save_mask=mask,
        extra=extra,
    )

    if target_sparsity is None:
        return candidate

    cur_sparsity = _get_sparsity(candidate, hidden_size)
    while cur_sparsity < target_sparsity:
        mask = _sampling_by_weight(
            d_target_sparsity,
            hidden_size,
            size,
            method,
        )

        # Get true field dims per fields
        candidate = DeepFMCandidate(
            save_mask=mask,
            extra=extra,
        )

        cur_sparsity = _get_sparsity(candidate, hidden_size)
    return candidate


def _validate_candidate(
    model,
    candidate: Candidate,
    val_loader,
):
    # this import here because it kind of weird for OptEmbed
    # to be depends on trainer

    from src.trainer.deepfm import validate_epoch

    # hook
    # model.embedding.forward = partial(
    #     model.embedding.forward,
    #     mask_d=candidate.save_mask,
    # )
    model.eval()
    model.embedding.get_weight(candidate.save_mask)
    # validate
    result = validate_epoch(val_loader, model)

    # unhook
    # model.embedding.forward = model.embedding.forward.func
    return result["auc"]


def _crossover(
    cur_top_candidate: List[Candidate],
    n_crossover: int,
    hidden_size,
    target_sparsity: Optional[float] = None,
) -> List[Candidate]:
    result = []
    for _ in range(n_crossover):
        while True:
            father, mother = random.choices(cur_top_candidate, k=2)

            # mix save mask
            father_mask = father.save_mask
            mother_mask = mother.save_mask
            son_mask = torch.empty_like(father_mask)
            father_choice_mask = torch.randint(2, size=(len(father_mask),), dtype=bool)
            mother_choice_mask = torch.logical_not(father_choice_mask)
            son_mask[father_choice_mask] = father_mask[father_choice_mask]
            son_mask[mother_choice_mask] = mother_mask[mother_choice_mask]

            candidate = Candidate(son_mask, father[1])

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
    hidden_size: int,
    target_sparsity: Optional[float] = None,
    d_target_sparsity: Optional[float] = None,
    method=1,
) -> List[Candidate]:
    result = []

    if target_sparsity is not None and d_target_sparsity is None:
        d_target_sparsity = target_sparsity

    max_hidden_size_budget = hidden_size
    for _ in range(n_mutate):
        while True:
            parent = random.choice(cur_top_candidate)

            son_mask = parent.save_mask.clone()
            mask = torch.rand(son_mask.shape[0]) < p_mutate
            num_mutated = mask.sum().item()
            son_mask[mask] = _sampling_by_weight(
                d_target_sparsity,
                hidden_size,
                num_mutated,
                method,
            )

            candidate = Candidate(son_mask, parent[1])

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
    sub_mask, n_max_elements = candidate.extra
    n_elements = ((candidate.save_mask + 1) * sub_mask).sum()

    cur_sparsity = 1 - n_elements / n_max_elements
    return cur_sparsity


def evol_search_deepfm(
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
    method=1,
) -> Tuple[torch.LongTensor, torch.LongTensor, float]:
    """Evolutionary search for DeepFM with OptEmbed

    Args:
        model: DeepFM model with OptEmbed
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
        method: Generate candidate with target sparsity
            0: Uniform (original in paper)
            1: exponential
            2: linear

    Returns:
        best_mask: (torch.LongTensor, shape (num_items,))
            best_mask[i] = how many dimension assigned for field/feature i

        best_auc (float)
    """

    cur_top_values = None
    cur_top_candidate = []

    if torch.cuda.is_available():
        model = model.cuda()

    assert isinstance(model.embedding, IOptEmbed)
    hidden_size = model.embedding._hidden_size

    with torch.no_grad():
        sub_mask = model.embedding.get_submask()

    d_target_sparsity = None
    if target_sparsity is not None:
        cur_ele_percent = sub_mask.sum() / model.embedding._num_item
        d_target_sparsity = 1 - (1 - target_sparsity) / cur_ele_percent

    logger.debug(f"{d_target_sparsity=}")
    candidates = [
        _generate_candidate(
            model.embedding,
            target_sparsity,
            d_target_sparsity,
            method,
        )
        for _ in range(population)
    ]

    for gen in range(n_generations):
        logger.debug(f"start {gen=}")
        metrics = torch.tensor(
            [
                _validate_candidate(model, candidate, val_dataloader)
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
            f"- auc: {cur_top_values[0]:.4f}"
            f"- sparsity: {cur_best_sparsity:.4f}"
        )

        if gen != n_generations - 1:
            logger.debug(f"mutate and crossover {gen=}")
            candidates = []
            crossovers = _crossover(
                cur_top_candidate,
                n_crossover,
                hidden_size,
                target_sparsity,
            )
            candidates.extend(crossovers)

            mutates = _mutate(
                cur_top_candidate,
                n_mutate,
                p_mutate,
                hidden_size,
                target_sparsity,
                d_target_sparsity,
                method,
            )
            candidates.extend(mutates)

    top_candidate = cur_top_candidate[0]
    mask = top_candidate.save_mask

    return mask, cur_top_values[0]


def _delete_cache(module: OptEmbed, grad_input, grad_output):
    module._cur_weight = None

    if isinstance(module, OptEmbed):
        module.get_submask.cache_clear()


# Retrain
class RetrainOptEmbed(IOptEmbed):
    def __init__(
        self,
        field_dims: Union[List[int], int],
        hidden_size,
        mode: Optional[str] = None,
        t_init: Optional[float] = 0,
        mode_threshold_e="field",
        mode_threshold_d="field",
        norm=1,
        target_sparsity: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        num_item = sum(field_dims)
        self._num_item = num_item
        self._field_dims = torch.tensor(field_dims, dtype=torch.int64)
        self._mode_d = mode_threshold_d
        self._mode = mode
        self._hidden_size = hidden_size

        self._weight = nn.Parameter(torch.empty((self._num_item, hidden_size)))
        _full_mask_d = get_mask(hidden_size)
        self.register_buffer("_full_mask_d", _full_mask_d)

        self._mask_d = None
        self._mask_e = None
        self._mask = None
        self._sparsity = 0
        self._cur_weight = None

    def init_mask(self, mask_e, mask_d):
        """Calculate final mask to apply to create weight"""

        # Get mask e
        mask_e = mask_e.unsqueeze(-1)

        # Get mask d
        if self._mode_d == "field":
            mask_d = torch.repeat_interleave(
                mask_d,
                self._field_dims,
                dim=0,
                output_size=self._num_item,
            )
        mask_d = F.embedding(mask_d, self._full_mask_d)

        # Apply mask
        self._mask = nn.Parameter(mask_d * mask_e, False)

        self._cur_weight = None
        self._handle = self.register_full_backward_hook(_delete_cache)

        logger.info(f"Params: {torch.nonzero(self._mask).size(0)}")

        return self._mask

    def get_weight(self, mask_d: Optional[torch.Tensor] = None):
        assert self._mask is not None, "Mask is not initialized"

        self._cur_weight = self._weight * self._mask
        return self._cur_weight

    def forward(self, x, mask_d=None):
        if self._cur_weight is None:
            self.get_weight()

        if self._mode is None:
            return F.embedding(x, self._cur_weight)
        else:
            return F.embedding_bag(x, self._cur_weight, mode=self._mode)

    def get_sparsity(self, get_n_params=False):
        nnz = torch.nonzero(self._mask).size(0)
        sparsity = 1 - nnz / (self._hidden_size * self._num_item)

        if not get_n_params:
            return sparsity
        return sparsity, nnz

    def get_num_params(self):
        nnz = torch.nonzero(self._mask).size(0)
        return nnz
