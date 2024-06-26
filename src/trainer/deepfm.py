import datetime
from typing import Dict, List, Tuple, Union, cast

import loguru
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.deepfm import DeepFM

now = datetime.datetime.now

# first is feat (without offset), second is label
CTR_DATA = Tuple[torch.Tensor, float]


def train_epoch(
    dataloader: DataLoader[CTR_DATA],
    model: DeepFM,
    optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    device="cuda",
    log_step=10,
    profiler=None,
    clip_grad=0,
) -> Dict[str, float]:
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    model.train()
    model.to(device)

    loss_dict: Dict[str, float] = dict(loss=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    load_data_time = datetime.timedelta()
    train_time = datetime.timedelta()
    first_start = start = now()

    for idx, batch in enumerate(dataloader):
        load_data_time += now() - start

        start_train = now()
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        for optimizer in optimizers:
            optimizer.step()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            loguru.logger.info(msg)

        if profiler:
            profiler.step()

        end_train = start = now()
        train_time += end_train - start_train

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    loguru.logger.info(f"train_time: {train_time}")
    loguru.logger.info(f"load_data_time: {load_data_time}")

    total_time = now() - first_start
    loguru.logger.info(f"total_time: {total_time}")

    return loss_dict


@torch.no_grad()
def validate_epoch(
    val_loader: DataLoader[CTR_DATA],
    model: DeepFM,
    device="cuda",
) -> Dict[str, float]:
    """Validate single epoch performance

    Args:
        val_dataloader
        model
        device
    Returns:
        "auc"
        "log_loss"
    """

    model.eval()
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    criterion = criterion.to(device)

    log_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for idx, batch in enumerate(val_loader):
        inputs, labels = batch
        all_y_true.extend(labels.tolist())

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        log_loss += criterion(outputs, labels.float()).item()

        outputs = torch.sigmoid(outputs)
        all_y_pred.extend(outputs.cpu().tolist())

    auc = roc_auc_score(all_y_true, all_y_pred)
    log_loss = log_loss / len(all_y_pred)
    return {
        "auc": auc,
        "log_loss": log_loss,
    }


def train_epoch_cerp(
    dataloader: DataLoader,
    model: DeepFM,
    optimizer,
    device="cuda",
    log_step=10,
    profiler=None,
    clip_grad=0,
    target_sparsity=0.8,
    prune_loss_weight=0,
) -> Dict[str, float]:
    """

    Difference compare to original LightGCN training:
        - Add prune loss
        - Check sparsity per log step

    """
    from src.models.embeddings.cerp_embedding import CerpEmbedding

    model.train()
    model.to(device)

    loss_dict: Dict[str, float] = dict(
        loss=0,
        prune_loss=0,
        log_loss=0,
        sparsity=0,
        num_params=0,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    load_data_time = datetime.timedelta()
    train_time = datetime.timedelta()
    first_start = start = now()
    model.embedding = cast(CerpEmbedding, model.embedding)

    for idx, batch in enumerate(dataloader):
        load_data_time += now() - start

        start_train = now()
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        log_loss = criterion(outputs, labels.float())
        prune_loss = model.embedding.get_prune_loss()
        loss = log_loss + prune_loss_weight * prune_loss

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        loss_dict["log_loss"] += log_loss.item()
        loss_dict["prune_loss"] += prune_loss.item()
        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            # check sparsity
            sparsity, num_params = model.embedding.get_sparsity(get_n_params=True)
            loss_dict["sparsity"] = sparsity
            loss_dict["num_params"] = num_params

            for metric, value in loss_dict.items():
                if metric == "sparsity":
                    msg += f" - {metric}: {value:.4}"
                elif metric == "num_params":
                    msg += f" - {metric}: {value}"
                elif value != 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            loguru.logger.info(msg)
            if sparsity >= target_sparsity:
                return loss_dict

        if profiler:
            profiler.step()

        end_train = start = now()
        train_time += end_train - start_train

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    # check sparsity
    sparsity, num_params = model.embedding.get_sparsity(get_n_params=True)
    loss_dict["sparsity"] = sparsity
    loss_dict["num_params"] = num_params

    loguru.logger.info(f"train_time: {train_time}")
    loguru.logger.info(f"load_data_time: {load_data_time}")

    total_time = now() - first_start
    loguru.logger.info(f"total_time: {total_time}")

    return loss_dict
