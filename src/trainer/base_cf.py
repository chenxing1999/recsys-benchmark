import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from src.dataset.cf_graph_dataset import CFGraphDataset
from src.models import save_cf_emb_checkpoint
from src.models.embeddings import detect_special


class CFTrainer:
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
        self.config = config
        self.model_config = config["model"]
        self.mode, self.is_retrain = detect_special(config)

        self.early_stop_count = 0
        self.early_stop_config = config.get("early_stop_patience", 0)
        self.warmup = config.get("warmup", 0)
        self.best_ndcg = 0

        self.num_users = num_users
        self.num_items = num_items

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device

    @abstractmethod
    def train_epoch(self, dataloader, epoch_idx: int) -> Dict[str, float]:
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

    def epoch_end(self, train_metrics, val_metrics, epoch_idx) -> bool:
        """Returns if stop training or not"""

        config = self.config

        # Check early stop + save on best model
        if self.best_ndcg < val_metrics["ndcg"]:
            logger.info("New best, saving model...")
            self.best_ndcg = val_metrics["ndcg"]

            checkpoint = {
                "state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "val_metrics": val_metrics,
                "num_users": self.num_users,
                "num_items": self.num_items,
            }
            torch.save(checkpoint, config["checkpoint_path"])
            self.early_stop_count = 0
        elif self.warmup <= epoch_idx:
            self.early_stop_count += 1
            logger.debug(f"{self.early_stop_count=}")

            early_stop_config = self.early_stop_config
            if early_stop_config and self.early_stop_count > early_stop_config:
                return True

        if not self.is_special:
            return False

        # special case for is_cerp and is_pep
        cur_sparsity = train_metrics["sparsity"]
        if self.is_cerp or self.is_pep:
            # Check in list target sparsity
            if self.is_cerp:
                emb_conf = self.config["cerp"]
            elif self.is_pep:
                emb_conf = self.config["pep_config"]

            checkpoint_dir = emb_conf["trial_checkpoint"]
            main_target_sparsity = emb_conf["target_sparsity"]

            if cur_sparsity > main_target_sparsity:
                logger.info(f"Found main target sparsity. Save at {checkpoint_dir}")
                save_cf_emb_checkpoint(
                    self.model,
                    checkpoint_dir,
                    "target",
                )
                return True

        # this is for keeping with original API format
        if self.is_pep:
            stop = self._pep_train_callback(cur_sparsity)
            if stop:
                return True

        return False

    @property
    def is_opt_embed_d(self):
        return self.mode == "optembed_d"

    @property
    def is_opt_embed(self):
        return self.mode == "opt_embed"

    @property
    def is_pep(self):
        return self.mode == "pep"

    @property
    def is_cerp(self):
        return self.mode == "cerp"

    @property
    def is_special(self):
        """most retrain can be train normally, so let just define this"""
        if self.is_opt_embed_d:
            return False
        return self.mode is not None and not self.is_retrain

    def _init_not_retrain(self):
        """Model init logic for special not retrain mode"""
        if self.is_opt_embed_d:
            self._init_optembed_d()
            self._init_optimizer()
        elif self.is_pep:
            self._init_pep()
        elif self.is_cerp:
            self._init_cerp()
        else:
            raise NotImplementedError(f"{self.mode=} - {self.is_retrain=}")

    def _init_pep(self):
        """Initialize custom optimizer and save init model"""

        # Initialize custom optimizer for PEP
        pep_config = self.config["pep_config"]
        model_weight_decay = 0
        use_warmup = pep_config.get("use_warmup", False)
        if not use_warmup:
            model_weight_decay = pep_config.get("model_weight_decay", 0)

        ps = [
            # threshold
            {"params": [], "weight_decay": pep_config["weight_decay"]},
            # non-threshold
            {
                "params": [],
                "weight_decay": model_weight_decay,
            },
        ]
        for name, p in self.model.named_parameters():
            if name.endswith(".s"):
                ps[0]["params"].append(p)
            else:
                ps[1]["params"].append(p)

        self.optimizer = torch.optim.Adam(
            ps,
            lr=self.config["learning_rate"],
        )
        logger.info("Special Optimizer for PEP")
        logger.debug(self.optimizer)

        # Save model_init
        model_init_path = pep_config["model_init_path"]
        if os.path.exists(model_init_path):
            logger.debug(f"init available model at {model_init_path}")
            state = torch.load(model_init_path)
            self.model.load_state_dict(state, strict=False)
        else:
            logger.debug(f"save init model to {model_init_path}")
            dirname = os.path.dirname(model_init_path)
            os.makedirs(dirname, exist_ok=True)
            state = self.model.state_dict()
            torch.save(state, model_init_path)

        emb_config = self.model_config["embedding_config"]
        target_sparsity = emb_config.get("target_sparsity")
        if not target_sparsity:
            target_sparsity = [0.5, 0.8, 0.95]

        target_sparsity.sort()
        self.target_sparsity = target_sparsity

    def _init_optembed_d(self):
        """Save init model"""
        config = self.config

        init_weight_path = config["opt_embed"]["init_weight_path"]
        torch.save(
            {
                "full": self.model.state_dict(),
            },
            init_weight_path,
        )

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )

    def _init_cerp(self):
        config = self.config
        weight_decay_threshold = config["cerp"]["weight_decay"]
        weight_decay_model = config["cerp"].get(
            "model_weight_decay", weight_decay_threshold
        )
        params = [
            # threshold
            {"weight_decay": weight_decay_threshold, "params": []},
            # normal weight
            {"weight_decay": weight_decay_model, "params": []},
        ]
        for name, p in self.model.named_parameters():
            if "threshold" in name:
                params[0]["params"].append(p)
            else:
                params[1]["params"].append(p)

        self.optimizer = torch.optim.Adam(
            params,
            lr=config["learning_rate"],
        )
        logger.debug(f"Special optimizer for CERP: {self.optimizer}")

        # Save initial checkpoint
        cerp_config = config["cerp"]
        trial_checkpoint = cerp_config["trial_checkpoint"]
        save_cf_emb_checkpoint(
            self.model,
            trial_checkpoint,
            "initial",
        )

    def _pep_train_callback(self, cur_sparsity: float) -> bool:
        """Code mimic logic for train_callback function of PEP

        TLDR, check if cur_sparsity > any sparsity current in list
            then update list

        """
        target_sparsity = self.target_sparsity
        emb_config = self.model_config["embedding_config"]

        cur_min_idx = 0

        checkpoint_dir = emb_config.get(
            "checkpoint_weight_dir",
            "checkpoints",
        )
        while (
            cur_min_idx < len(target_sparsity)
            and target_sparsity[cur_min_idx] < cur_sparsity
        ):
            sparsity = target_sparsity[cur_min_idx]
            logger.info(f"cur_sparsity is larger than {sparsity}")

            # Save model
            save_cf_emb_checkpoint(self.model, checkpoint_dir, f"{sparsity:.4f}")
            cur_min_idx += 1

        # update list
        leftover = len(target_sparsity) - cur_min_idx
        for i in range(leftover):
            target_sparsity[i] = target_sparsity[i + cur_min_idx]

        if cur_min_idx > 0:
            target_sparsity = target_sparsity[:leftover]
            self.target_sparsity = target_sparsity
            logger.debug(f"new {target_sparsity=}")

        if len(target_sparsity) == 0:
            return True
        return False

    def __repr__(self):
        emb_name = self.get_emb_name()
        return (
            f"{self.__class__.__name__}("
            f"mode={self.mode}, "
            f"is_retrain={self.is_retrain}, "
            f"emb_name={emb_name}"
            ")"
        )

    def get_emb_name(self):
        return self.config["model"].get("embedding_config", {"name": "vanilla"})["name"]
