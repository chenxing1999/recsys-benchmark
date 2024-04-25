import random
from dataclasses import dataclass
from typing import List

import torch

from src.models.embeddings.lightgcn_opt_embed import OptEmbed
from src.models.embeddings.optembed_evol_base import EvolSearchOpt
from src.models.mlp import NeuMF
from src.trainer.nmf import validate_epoch

from .optembed_utils import _sampling_by_weight


@dataclass
class NmfCandidate:
    mlp_user: torch.Tensor
    mlp_item: torch.Tensor
    gmf_user: torch.Tensor
    gmf_item: torch.Tensor

    @property
    def user_mask(self):
        return torch.cat([self.mlp_user, self.gmf_user])

    @property
    def item_mask(self):
        return torch.cat([self.mlp_item, self.gmf_item])


class NmfSearchOpt(EvolSearchOpt[NmfCandidate]):
    model: NeuMF

    STEP = 1.05

    def __init__(self, *args, **kwargs):
        from typing import cast

        super().__init__(*args, **kwargs)
        self.model = cast(NeuMF, self.model)

        self.device = "cuda"
        self.model.eval()
        self.model = self.model.to(self.device)

    def _get_hidden_size(self):
        return self.model._emb_size // 2

    def _generate_candidate(
        self,
        num_users,
        num_items,
        hidden_size,
    ) -> NmfCandidate:
        candidate = NmfCandidate(
            gmf_user=_sampling_by_weight(
                self.target_sparsity,
                self._get_hidden_size(),
                num_users,
                self.method,
            ),
            gmf_item=_sampling_by_weight(
                self.target_sparsity,
                self._get_hidden_size(),
                num_items,
                self.method,
            ),
            mlp_user=_sampling_by_weight(
                self.target_sparsity,
                self._get_hidden_size(),
                num_users,
                self.method,
            ),
            mlp_item=_sampling_by_weight(
                self.target_sparsity,
                self._get_hidden_size(),
                num_items,
                self.method,
            ),
        )
        if self.target_sparsity is None:
            return candidate

        cur_sparsity = self._get_sparsity(candidate, hidden_size)
        while cur_sparsity < self.target_sparsity:
            candidate = NmfCandidate(
                gmf_user=_sampling_by_weight(
                    self.target_sparsity * self.STEP,
                    self._get_hidden_size(),
                    num_users,
                    self.method,
                ),
                gmf_item=_sampling_by_weight(
                    self.target_sparsity * self.STEP,
                    self._get_hidden_size(),
                    num_items,
                    self.method,
                ),
                mlp_user=_sampling_by_weight(
                    self.target_sparsity * self.STEP,
                    self._get_hidden_size(),
                    num_users,
                    self.method,
                ),
                mlp_item=_sampling_by_weight(
                    self.target_sparsity * self.STEP,
                    self._get_hidden_size(),
                    num_items,
                    self.method,
                ),
            )
            cur_sparsity = self._get_sparsity(candidate, hidden_size)
        return candidate

    def _validate_candidate(
        self,
        candidate: NmfCandidate,
        val_dataloader,
        train_dataset,
        metrics=None,
    ) -> float:
        model = self.model
        model.eval()

        self._set_weight(
            model._mlp.user_emb_table,
            candidate.mlp_user,
        )

        self._set_weight(
            model._mlp.item_emb_table,
            candidate.mlp_item,
        )

        self._set_weight(
            model._gmf.user_emb_table,
            candidate.gmf_user,
        )

        self._set_weight(
            model._gmf.item_emb_table,
            candidate.gmf_item,
        )
        result = validate_epoch(
            train_dataset, val_dataloader, self.model, metrics=metrics
        )
        if metrics is None:
            return result["ndcg"]
        return result

    def _set_weight(self, emb: OptEmbed, mask: torch.Tensor) -> OptEmbed:
        """
        Args:
            emb:
        """
        emb._cur_weight = emb.get_weight(mask)
        return emb

    def _get_sparsity(self, candidate, hidden_size) -> float:
        num_params = (candidate.mlp_user + 1).sum()
        num_params += (candidate.mlp_item + 1).sum()
        num_params += (candidate.gmf_user + 1).sum()
        num_params += (candidate.gmf_item + 1).sum()

        total_params = (
            (len(candidate.mlp_item) + len(candidate.mlp_user)) * hidden_size * 2
        )

        return 1 - num_params / total_params

    def _get_num_params(self, candidate):
        num_params = (candidate.mlp_user + 1).sum()
        num_params += (candidate.mlp_item + 1).sum()
        num_params += (candidate.gmf_user + 1).sum()
        num_params += (candidate.gmf_item + 1).sum()
        return num_params

    def _mutate(
        self,
        cur_top_candidate: List[NmfCandidate],
        num_users: int,
        num_items: int,
        hidden_size: int,
    ) -> List[NmfCandidate]:
        max_hidden_size_budget = hidden_size

        target_sparsity = self.target_sparsity
        p_mutate = self.p_mutate

        result = []

        for _ in range(self.n_mutate):
            while True:
                parent: NmfCandidate = random.choice(cur_top_candidate)

                son_item = parent.item_mask
                mask = torch.rand(num_items * 2) < p_mutate
                num_mutated = mask.sum().item()
                son_item[mask] = _sampling_by_weight(
                    target_sparsity,
                    hidden_size,
                    num_mutated,
                    self.method,
                )

                son_user = parent.user_mask
                mask = torch.rand(num_users * 2) < p_mutate
                num_mutated = mask.sum().item()
                son_user[mask] = _sampling_by_weight(
                    target_sparsity,
                    hidden_size,
                    num_mutated,
                    self.method,
                )

                candidate = NmfCandidate(
                    mlp_item=son_item[:num_items],
                    gmf_item=son_item[num_items:],
                    mlp_user=son_user[:num_users],
                    gmf_user=son_user[num_users:],
                )

                if target_sparsity is None:
                    break

                cur_sparsity = self._get_sparsity(candidate, hidden_size)
                if cur_sparsity > target_sparsity:
                    break

                max_hidden_size_budget -= 1

            result.append(candidate)

        return result

    def _crossover(
        self, cur_top_candidate: list, num_users, num_items, hidden_size
    ) -> List[NmfCandidate]:
        target_sparsity = self.target_sparsity
        result = []

        for _ in range(self.n_crossover):
            while True:
                father, mother = random.choices(cur_top_candidate, k=2)

                father: NmfCandidate
                mother: NmfCandidate

                # mix user
                father_user = father.user_mask
                mother_user = mother.user_mask
                son_user = torch.empty_like(father_user)
                father_choice_mask = torch.randint(
                    2, size=(son_user.shape[0],), dtype=bool
                )
                mother_choice_mask = torch.logical_not(father_choice_mask)
                son_user[father_choice_mask] = father_user[father_choice_mask]
                son_user[mother_choice_mask] = mother_user[mother_choice_mask]

                # mix item
                father_item = father.item_mask
                mother_item = mother.item_mask
                son_item = torch.empty_like(father_item)
                father_choice_mask = torch.randint(
                    2, size=(son_item.shape[0],), dtype=bool
                )
                mother_choice_mask = torch.logical_not(father_choice_mask)
                son_item[father_choice_mask] = father_item[father_choice_mask]
                son_item[mother_choice_mask] = mother_item[mother_choice_mask]

                candidate = NmfCandidate(
                    mlp_item=son_item[:num_items],
                    gmf_item=son_item[num_items:],
                    mlp_user=son_user[:num_users],
                    gmf_user=son_user[num_users:],
                )

                if target_sparsity is None:
                    break

                cur_sparsity = self._get_sparsity(candidate, hidden_size)
                if cur_sparsity > target_sparsity:
                    break

            result.append(candidate)
        return result

    def _extract(self, candidate) -> List[torch.Tensor]:
        return [
            candidate.mlp_user,
            candidate.mlp_item,
            candidate.gmf_user,
            candidate.gmf_item,
        ]

    def _load_candidate(self, init_weight_path):
        info = torch.load(init_weight_path)
        mask = info["mask"]

        candidate = NmfCandidate(
            mlp_user=mask["mlp_user"]["mask_d"],
            mlp_item=mask["mlp_item"]["mask_d"],
            gmf_user=mask["gmf_user"]["mask_d"],
            gmf_item=mask["gmf_item"]["mask_d"],
        )
        return candidate

    def validate_result(self, init_weight_path, val_loader, train_dataset):
        candidate = self._load_candidate(init_weight_path)
        val_metrics = self._validate_candidate(
            candidate,
            val_loader,
            train_dataset,
            metrics=["ndcg", "recall"],
        )

        val_metrics["num_params"] = self._get_num_params(candidate)
        return val_metrics
