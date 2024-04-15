"""Define training and evaluating logic for NeuMF"""
from typing import Any, Dict, List, Optional, Set, Union

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src import metrics as metric_utils
from src.dataset.cf_graph_dataset import CFGraphDataset
from src.models.mlp import ModelFlag, NeuMF, get_sparsity_and_param
from src.trainer.base_cf import CFTrainer


class NeuMFTrainer(CFTrainer):
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
        super().__init__(num_users, num_items, config)

        model_config = self.model_config.copy()
        model_config.pop("name")

        self._model = NeuMF(num_users, num_items, **model_config)
        self._model.flag = ModelFlag.NMF
        assert (
            self.mode != "optembed"
        ), "Not implemented for OptEmbed. Please use Legacy code"

        self.l2_reg = config["weight_decay"]

        self.pretrain_step = config.get("pretrain_step", 0)
        if self.pretrain_step:
            logger.debug(f"Enable pretrain {self.pretrain_step=}")
            self._model.flag = ModelFlag.MLP
            assert self.pretrain_step % 2 == 0

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
                if key
                in [
                    "_gmf.user_emb_table._mask",
                    "_mlp.user_emb_table._mask",
                    "_gmf.item_emb_table._mask",
                    "_mlp.item_emb_table._mask",
                ]
            )
            length_miss = length_miss - expected_miss
            assert length_miss == 0, f"There are some keys missing: {keys[0]}"
            self.model._mlp.item_emb_table.init_mask(
                mask_d=mask["mlp_item"]["mask_d"], mask_e=None
            )
            self.model._mlp.user_emb_table.init_mask(
                mask_d=mask["mlp_user"]["mask_d"], mask_e=None
            )
            self.model._gmf.item_emb_table.init_mask(
                mask_d=mask["gmf_item"]["mask_d"], mask_e=None
            )
            self.model._gmf.user_emb_table.init_mask(
                mask_d=mask["gmf_user"]["mask_d"], mask_e=None
            )

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
        if self.pretrain_step:
            if epoch_idx == self.pretrain_step // 2:
                self._model.flag = ModelFlag.GMF
            elif epoch_idx == self.pretrain_step:
                self._model.flag = ModelFlag.NMF
                self._model.update_weight(0.5)

        emb_name = self.get_emb_name()
        if emb_name == "tt_emb" and epoch_idx == 5:
            for name, emb in self.model.get_embs():
                emb.cache_populate()

        if not self.is_special or self.mode == "optembed_d":
            return train_epoch(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.l2_reg,  # l2 on embedding
                None,
            )

        if self.is_pep:
            return train_epoch_pep(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.l2_reg,  # l2 on embedding
                None,
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
            return train_epoch_cerp(
                dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.config["log_step"],
                self.l2_reg,  # l2 on embedding
                None,
                self.config["cerp"]["target_sparsity"],
                (gamma_decay**epoch_idx) * gamma_init,
            )

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
        self.model.eval()

        result = validate_epoch(
            train_dataset,
            dataloader,
            self.model,
            self.device,
            metrics=metrics,
        )

        self.model.clear_cache()
        return result

    @property
    def model(self):
        return self._model

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            # weight_decay=1e-3,
        )


def _log_loss(y_hat_pos, y_hat_neg):
    rec_loss = F.binary_cross_entropy_with_logits(
        y_hat_pos,
        torch.ones_like(y_hat_pos),
    )
    rec_loss += F.binary_cross_entropy_with_logits(
        y_hat_neg,
        torch.zeros_like(y_hat_neg),
    )
    return rec_loss


def train_epoch(
    dataloader: DataLoader,
    model: NeuMF,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
) -> Dict[str, float]:
    """ """
    model.train()
    model.to(device)

    num_sample = 0

    loss_dict: Dict[str, float] = dict(
        loss=0,
        reg_loss=0,
        rec_loss=0,
    )
    for idx, batch in enumerate(dataloader):
        loss, rec_loss, reg_loss = _train_step(
            batch, model, optimizer, device, weight_decay
        )

        loss_dict["loss"] += loss.item()
        loss_dict["rec_loss"] += rec_loss.item()
        loss_dict["reg_loss"] += reg_loss.item()

        num_sample += batch[0].shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            logger.info(msg)

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


def train_epoch_pep(
    dataloader: DataLoader,
    model: NeuMF,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    target_sparsity=0,
) -> Dict[str, float]:
    """Training Epoch for PEP Pretrain step with NeuMF model

    Args:

    Returns: Dict contains following keys:
        loss
        reg_loss
        rec_loss
        cl_loss
        sparsity
        num_params
    """
    model.train()
    model.to(device)

    num_sample = 0

    loss_dict: Dict[str, float] = dict(
        loss=0,
        reg_loss=0,
        rec_loss=0,
    )
    for idx, batch in enumerate(dataloader):
        loss, rec_loss, reg_loss = _train_step(
            batch, model, optimizer, device, weight_decay
        )

        loss_dict["loss"] += loss.item()
        loss_dict["rec_loss"] += rec_loss.item()
        loss_dict["reg_loss"] += reg_loss.item()

        num_sample += batch[0].shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            sparsity, num_params = get_sparsity_and_param(model)
            msg += f" - sparsity: {sparsity:.2f} - num_params: {num_params}"

            loss_dict["sparsity"] = sparsity
            loss_dict["num_params"] = num_params
            for metric, value in loss_dict.items():
                if metric in ["sparsity", "num_params"]:
                    continue
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            logger.info(msg)
            if sparsity > target_sparsity:
                logger.info("Found target sparsity")
                break

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        if metric in ["sparsity", "num_params"]:
            continue
        loss_dict[metric] = avg

    return loss_dict


def train_epoch_cerp(
    dataloader: DataLoader,
    model: NeuMF,
    optimizer,
    device="cuda",
    log_step=10,
    weight_decay=0,
    profiler=None,
    target_sparsity=0,
    prune_loss_weight=0,
    clip_grad_norm=100,
) -> Dict[str, float]:
    """Training Epoch for PEP Pretrain step with NeuMF model

    Args:

    Returns: Dict contains following keys:
        loss
        reg_loss
        rec_loss
        cl_loss
        sparsity
        num_params
    """
    model.train()
    model.to(device)

    num_sample = 0

    loss_dict: Dict[str, float] = dict(
        loss=0,
        reg_loss=0,
        rec_loss=0,
        prune_loss=0,
    )
    for idx, batch in enumerate(dataloader):
        loss, rec_loss, reg_loss, prune_loss = _train_step(
            batch,
            model,
            optimizer,
            device,
            weight_decay,
            prune_loss_weight,
            clip_grad_norm,
        )

        loss_dict["loss"] += loss.item()
        loss_dict["rec_loss"] += rec_loss.item()
        loss_dict["reg_loss"] += reg_loss.item()
        loss_dict["prune_loss"] += prune_loss.item()

        num_sample += batch[0].shape[0]

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            sparsity, num_params = get_sparsity_and_param(model)
            msg += f" - sparsity: {sparsity:.2f} - num_params: {num_params}"

            loss_dict["sparsity"] = sparsity
            loss_dict["num_params"] = num_params
            for metric, value in loss_dict.items():
                if metric in ["sparsity", "num_params"]:
                    continue
                if value > 0 or metric in ["prune_loss", "loss"]:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            logger.info(msg)
            if sparsity > target_sparsity:
                logger.info("Found target sparsity")
                break

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        if metric in ["sparsity", "num_params"]:
            continue
        loss_dict[metric] = avg

    return loss_dict


def _train_step(
    batch,
    model: NeuMF,
    optimizer,
    device="cuda",
    weight_decay=0,
    prune_loss_weight=0,
    clip_grad_norm=0,
):
    users, pos_items, neg_items = batch

    users = users.to(device)
    pos_items = pos_items.to(device)

    n_repeat = 1
    if isinstance(neg_items, list):
        n_repeat = len(neg_items)
        neg_items = torch.cat(neg_items)

    neg_items = neg_items.to(device)

    # y_hat_pos = model(users, pos_items)
    # y_hat_neg = model(users.repeat(n_repeat), neg_items)

    # work around batch norm of DHE
    y_hat = model(users.repeat(n_repeat + 1), torch.cat([pos_items, neg_items]))
    y_hat_pos = y_hat[: len(pos_items)]
    y_hat_neg = y_hat[len(pos_items) :]

    # bpr loss is actually slightly worse on a quick test
    # rec_loss = -F.logsigmoid(y_hat_pos - y_hat_neg).mean()
    rec_loss = _log_loss(y_hat_pos, y_hat_neg)

    reg_loss: torch.Tensor = torch.tensor(0)
    if weight_decay > 0:
        reg_loss = model.get_reg_loss(users, pos_items, neg_items)

    prune_loss = torch.tensor(0)
    if prune_loss_weight > 0:
        prune_loss = model.get_prune_loss_tanh(users, pos_items, neg_items)

    loss = rec_loss + weight_decay * reg_loss + prune_loss * prune_loss_weight
    optimizer.zero_grad()
    loss.backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()

    if prune_loss_weight > 0:
        return loss, rec_loss, reg_loss, prune_loss

    return loss, rec_loss, reg_loss


@torch.no_grad()
def validate_epoch(
    train_dataset: CFGraphDataset,
    val_loader: DataLoader,
    model: NeuMF,
    device="cuda",
    k=20,
    filter_item_on_train=True,
    profiler=None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Validate single epoch performance

    Args:
        train_dataset
           For getting num_users, norm_adj and filter interacted item

        val_dataloader
        model
        device
        k
        filter_item_on_train: Remove item that user already interacted on train

        metrics: Only support `ndcg` and `recall`
    Returns:
        "ndcg"
    """
    graph = train_dataset.get_graph()

    model.eval()

    model = model.to(device)
    num_items = model.num_item

    all_items = torch.arange(num_items, device=device).unsqueeze(0)

    ndcg: float = 0
    all_y_pred = []
    all_y_true = []

    pos_items: List[Union[Set[int], List[int]]]
    for users, pos_items in val_loader:
        # scores: batch_user x num_items
        batch_size = users.shape[0]
        user_tensor = users.to(device).unsqueeze(1).repeat(1, num_items)
        scores = model(
            user_tensor,
            all_items.repeat(batch_size, 1),
        )

        if filter_item_on_train:
            ind0 = []
            ind1 = []
            for idx, user in enumerate(users.tolist()):
                ind0.extend([idx] * len(graph[user]))
                ind1.extend(graph[user])

            scores[ind0, ind1] = float("-inf")

        y_pred = torch.topk(scores, k)
        y_pred = y_pred[1]

        all_y_pred.extend(y_pred.cpu().tolist())

        all_y_true.extend(pos_items)
        if profiler:
            profiler.step()

    if metrics is None:
        ndcg = metric_utils.get_ndcg(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
        }
    elif "ndcg" in metrics and "recall" in metrics:
        ndcg, recall = metric_utils.get_ndcg_recall(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
            "recall": recall,
        }
    else:
        ndcg = metric_utils.get_ndcg(all_y_pred, all_y_true, k)
        return {
            "ndcg": ndcg,
        }


if __name__ == "__main__":
    import yaml

    from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset

    def get_config(config_file) -> Dict:
        with open(config_file) as fin:
            config = yaml.safe_load(fin)
        return config

    config = get_config(
        "/home/xing/workspace/phd/recsys-benchmark/configs/gowalla/base_config.yaml"
    )
    # logger = Logger(**config["logger"])

    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config, num_neg_item=1)
    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataloader_config["batch_size"],
        shuffle=True,
        num_workers=train_dataloader_config["num_workers"],
    )

    logger.info("Load val dataset...")
    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]
    val_dataset = TestCFGraphDataset(val_dataloader_config["dataset"]["path"])
    val_dataloader = DataLoader(
        val_dataset,
        32,
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=val_dataloader_config["num_workers"],
    )
    logger.info("Successfully load val dataset")

    model = NeuMF(
        train_dataset.num_users,
        train_dataset.num_items,
        emb_size=64,
        hidden_sizes=[64, 32, 16],
        p_dropout=0.2,
        # embedding_config={"name": "vanilla", "initializer": "default"},
    )
    # 32 + 8 + 16 = 15

    optimizer = torch.optim.Adam(
        model.parameters(),
        1e-2,
        # weight_decay=1e-4,
    )

    from pprint import pprint

    best_ndcg = 0
    early_stop = 0

    model.flag = ModelFlag.MLP
    num_warmup = 0
    for i in range(100):
        if i == num_warmup:
            model.update_weight(0.5)
            model.flag = ModelFlag.NMF
        elif i == num_warmup // 2:
            model.flag = ModelFlag.GMF

        print(f"--epoch {i}--")
        print("---training---")
        pprint(
            train_epoch(
                train_dataloader,
                model,
                optimizer,
                log_step=100,
                weight_decay=1e-3,
            )
        )

        print("---validate---")
        val_res = validate_epoch(train_dataset, val_dataloader, model, k=10)
        pprint(val_res)
        ndcg = val_res["ndcg"]
        if best_ndcg < ndcg:
            best_ndcg = ndcg
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 5:
                break
    print(best_ndcg)
