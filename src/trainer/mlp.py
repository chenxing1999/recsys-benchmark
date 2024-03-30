"""Define training and evaluating logic for MLP"""
from typing import Dict, List, Optional, Set, Union

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src import metrics as metric_utils
from src.dataset.cf_graph_dataset import CFGraphDataset
from src.models.mlp import MLP_CF


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
    model: MLP_CF,
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
        users, pos_items, neg_items = batch

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        y_hat_pos = model(users, pos_items)
        y_hat_neg = model(users, neg_items)

        # bpr loss is actually slightly worse on a quick test
        # rec_loss = -F.logsigmoid(y_hat_pos - y_hat_neg).mean()
        rec_loss = _log_loss(y_hat_pos, y_hat_neg)

        reg_loss: torch.Tensor = torch.tensor(0)
        if weight_decay > 0:
            reg_loss = model.get_reg_loss(users, pos_items, neg_items)
            loss_dict["reg_loss"] += reg_loss.item()

        loss = rec_loss + weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict["loss"] += loss.item()
        loss_dict["rec_loss"] += rec_loss.item()

        num_sample += users.shape[0]

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


@torch.no_grad()
def validate_epoch(
    train_dataset: CFGraphDataset,
    val_loader: DataLoader,
    model: MLP_CF,
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
    model.num_user

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
    from src.loggers import Logger

    def get_config(config_file) -> Dict:
        with open(config_file) as fin:
            config = yaml.safe_load(fin)
        return config

    config = get_config(
        "/home/xing/workspace/phd/recsys-benchmark/configs/gowalla/base_config.yaml"
    )
    logger = Logger(**config["logger"])

    logger.info("Load train dataset...")
    train_dataloader_config = config["train_dataloader"]
    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = CFGraphDataset(**train_dataset_config)
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

    model = MLP_CF(
        train_dataset.num_users,
        train_dataset.num_items,
        emb_size=64,
        hidden_sizes=[128, 256, 512],
        p_dropout=0.5,
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
    for i in range(100):
        print(f"--epoch {i}--")
        print("---training---")
        pprint(
            train_epoch(
                train_dataloader,
                model,
                optimizer,
                log_step=100,
                # weight_decay=1e-3,
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
