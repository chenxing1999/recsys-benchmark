from typing import Generic, List, Tuple, TypeVar

import torch
from loguru import logger

Candidate = TypeVar("Candidate")


class EvolSearchOpt(Generic[Candidate]):
    def __init__(
        self,
        model,
        n_generations: int,
        population: int,
        n_crossover: int,
        n_mutate: int,
        p_mutate: float,
        k: int,
        target_sparsity=None,
        method=1,
    ):
        self.n_generations = n_generations
        self.population = population
        self.n_crossover = n_crossover
        self.n_mutate = n_mutate
        self.p_mutate = p_mutate
        self.k = k
        self.target_sparsity = target_sparsity
        self.method = method
        self.model = model

    def _get_hidden_size(self):
        ...

    def _generate_candidate(
        self,
        num_users,
        num_items,
        hidden_size,
    ) -> Candidate:
        ...

    def _validate_candidate(
        self,
        candidate,
        val_dataloader,
        train_dataset,
    ) -> float:
        ...

    def _get_sparsity(self, candidate, hidden_size):
        ...

    def _mutate(
        self,
        cur_top_candidate: List[Candidate],
        num_users: int,
        num_items: int,
        hidden_size: int,
    ) -> List[Candidate]:
        ...

    def _crossover(
        self, cur_top_candidate: list, num_users, num_items, hidden_size
    ) -> List[Candidate]:
        ...

    def _extract(self, candidate) -> List[torch.Tensor]:
        ...

    def evol_search(
        self,
        val_dataloader,
        train_dataset,
    ) -> Tuple[List[torch.Tensor], float]:
        """Evolutionary search

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
            method: Generate candidate with target sparsity
                0: Uniform (original in paper)
                1: exponential
                2: linear

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

        hidden_size = self._get_hidden_size()

        candidates = [
            self._generate_candidate(
                num_users,
                num_items,
                hidden_size,
            )
            for _ in range(self.population)
        ]

        for gen in range(self.n_generations):
            logger.debug(f"start {gen=}")
            metrics = torch.tensor(
                [
                    self._validate_candidate(candidate, val_dataloader, train_dataset)
                    for candidate in candidates
                ]
            )
            if cur_top_values is not None:
                cur_top_values = torch.cat((cur_top_values, metrics))
            else:
                cur_top_values = metrics
            cur_top_candidate.extend(candidates)

            result = torch.topk(cur_top_values, self.k)

            cur_top_candidate = [cur_top_candidate[idx] for idx in result.indices]
            cur_top_values = result.values

            cur_best_sparsity = self._get_sparsity(cur_top_candidate[0], hidden_size)
            logger.debug(
                f"cur best {cur_top_candidate[0]}"
                f"- best_metric: {cur_top_values[0]:.4f}"
                f"- sparsity: {cur_best_sparsity:.4f}"
            )

            if gen != self.n_generations - 1:
                logger.debug(f"mutate and crossover {gen=}")
                candidates = []
                crossovers = self._crossover(
                    cur_top_candidate,
                    num_users,
                    num_items,
                    hidden_size,
                )
                candidates.extend(crossovers)

                mutates = self._mutate(
                    cur_top_candidate,
                    num_users,
                    num_items,
                    hidden_size,
                )
                candidates.extend(mutates)

        # item_mask = top_candidate.item_mask
        # user_mask = top_candidate.user_mask

        # return item_mask, user_mask, cur_top_values[0]
        return self._extract(cur_top_candidate[0]), cur_top_values[0]
