from typing import Dict

import loguru
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.deepfm import DeepFM


def train_epoch(
    dataloader: DataLoader,
    model: DeepFM,
    optimizer,
    device="cuda",
    log_step=10,
    profiler=None,
) -> Dict[str, float]:
    model.train()
    model.to(device)

    loss_dict = dict(loss=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    for idx, batch in enumerate(dataloader):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.2}"

            loguru.logger.info(msg)

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


@torch.no_grad()
def validate_epoch(
    val_loader: DataLoader,
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

    log_loss = 0
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
