"""Re-implementation of LightGCN training logic into class
to fit everything into a single script"""
from typing import Any, Dict, List, Optional

import torch

from src.dataset.cf_graph_dataset import CFGraphDataset
from src.models import get_graph_model
from src.models.embeddings.cerp_embedding_utils import train_epoch_cerp
from src.trainer.base_cf import CFTrainer
from src.trainer.lightgcn import train_epoch, train_epoch_pep, validate_epoch


class GraphTrainer(CFTrainer):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: Dict[str, Any],
    ):
        super().__init__(num_users, num_items, config)

        assert (
            self.mode != "optembed"
        ), "Not implemented for OptEmbed. Please use Legacy code"

        model_config = config["model"]
        self._model = get_graph_model(num_users, num_items, model_config)

        # Emb specific code
        # note: will not support OptEmbed for now,
        # just support OptEmbed-D
        if self.is_retrain and self.is_opt_embed_d:
            self._init_optembed_d()
            self._init_optimizer()
        elif self.is_special:
            self._init_not_retrain()
        else:
            self._init_optimizer()

        # ----

    def _init_optembed_d(self):
        # opt_embed specific code
        config = self.config
        model_config = config["model"]
        is_retrain = "retrain" in model_config["embedding_config"]["name"]

        init_weight_path = config["opt_embed"]["init_weight_path"]
        if not is_retrain:
            torch.save(
                {
                    "full": self.model.state_dict(),
                },
                init_weight_path,
            )
        else:
            # if retrain, load original mask
            info = torch.load(init_weight_path)
            mask = info["mask"]
            keys = self.model.load_state_dict(info["full"], False)
            length_miss = len(keys[0])
            expected_miss = sum(
                1
                for key in keys[0]
                if key in ["user_emb_table._mask", "item_emb_table._mask"]
            )
            length_miss = length_miss - expected_miss
            assert length_miss == 0, f"There are some keys missing: {keys[0]}"

            # TODO: Refactor this later
            self.model.item_emb_table.init_mask(
                mask_d=mask["item"]["mask_d"], mask_e=None
            )
            self.model.user_emb_table.init_mask(
                mask_d=mask["user"]["mask_d"], mask_e=None
            )

    @property
    def model(self):
        return self._model

    def train_epoch(self, dataloader, epoch_idx: int) -> Dict[str, float]:
        # is retrain or just normal model
        if not self.is_special or self.mode == "optembed_d":
            return train_epoch(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.config["weight_decay"],
                None,
                self.config["info_nce_weight"],
            )

        if self.is_pep:
            return train_epoch_pep(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.config["weight_decay"],
                None,
                self.config["info_nce_weight"],
                self.config["pep_config"]["target_sparsity"],
            )

        elif self.is_opt_embed:
            raise NotImplementedError()
        elif self.is_cerp:
            cerp_config = self.config.get(
                "cerp", {"gamma_init": 1.0, "gamma_decay": 0.5}
            )
            gamma_init = cerp_config["gamma_init"]
            gamma_decay = cerp_config["gamma_decay"]
            target_sparsity = cerp_config["target_sparsity"]
            return train_epoch_cerp(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.config["weight_decay"],
                None,
                self.config["info_nce_weight"],
                prune_loss_weight=(gamma_decay**epoch_idx) * gamma_init,
                target_sparsity=target_sparsity,
            )

    def _train_epoch_pep(self, dataloader, epoch_idx):
        pep_config = self.config["pep_config"]
        use_warmup = self.config.get("use_warmup", False)
        model_weight_decay = pep_config.get("model_weight_decay", 0)

        if epoch_idx == 5 and use_warmup:
            self.optimizer.param_groups[1]["weight_decay"] = model_weight_decay

        # while training will continue to check sparsity
        # if statisfied sparsity: save
        return train_epoch_pep(
            dataloader,
            self.model,
            self.optimizer,
            self.device,
            self.config["log_step"],
            self.config["weight_decay"],
            None,
            self.config["info_nce_weight"],
            self.target_sparsity[-1],
        )

    def validate_epoch(
        self,
        train_dataset: CFGraphDataset,
        dataloader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        return validate_epoch(
            train_dataset,
            dataloader,
            self.model,
            self.device,
            metrics=metrics,
        )

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )
