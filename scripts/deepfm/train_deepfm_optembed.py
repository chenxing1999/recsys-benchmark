import argparse
import os
from typing import Dict, List, Optional, Sequence

import loguru
import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.criteo import get_dataset_cls
from src.loggers import Logger
from src.models.deepfm import DeepFM
from src.models.embeddings.deepfm_opt_embed import IOptEmbed, OptEmbed
from src.trainer.deepfm import validate_epoch
from src.utils import set_seed

set_seed(2023)


def train_epoch(
    dataloader: DataLoader,
    model: DeepFM,
    optimizers: List[torch.optim.Optimizer],
    device="cuda",
    log_step=10,
    profiler=None,
    clip_grad=0,
    alpha=0,
) -> Dict[str, float]:
    """Custom training logic for DeepFM OptEmbed

    Customization have been done:
        - Add `alpha` (weight for l_s)
        - Use multiple optimizers instead of one.
    """
    model.train()
    model.to(device)

    model.embedding: IOptEmbed

    loss_dict = dict(loss=0, loss_s=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    is_opt_embed = isinstance(model.embedding, OptEmbed)
    for idx, batch in enumerate(dataloader):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        loss_s = model.embedding.get_l_s()
        loss = loss + alpha * loss_s

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        for optimizer in optimizers:
            optimizer.step()

        if is_opt_embed:
            loss_dict["loss_s"] += loss_s.item()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"
            sparsity, n_params = model.embedding.get_sparsity(True)

            msg += f" - {sparsity=:.4f} - {n_params=}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            loguru.logger.info(msg)

            # DEBUG CODE ---
            # t_param = model.embedding._mask_e_module._t_param
            # print(
            #     f"Threshold --- "
            #     f"Max: {t_param.max()}"
            #     f"- Min: {t_param.min()}"
            #     f"- Mean: {t_param.mean()}"
            # )
            # norm = model.embedding._weight.norm(1, dim=1)
            # print(
            #     f"Norm --- "
            #     f"Max: {norm.max()}"
            #     f"- Min: {norm.min()}"
            #     f"- Mean: {norm.mean()}"
            # )

        if profiler:
            profiler.step()

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    return loss_dict


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def init_profiler(config: Dict):
    """Init PyTorch profiler based on config file

    Args:
        "log_path"
        "schedule"
            "wait"
            "warmup"
            "active"
            "repeat"
        "record_shapes" (default: False)
        "profile_memory" (default: True)
        "with_stack" (default: False)
    """

    log_path = config["log_path"]
    prof_schedule = torch.profiler.schedule(**config["schedule"])
    prof = torch.profiler.profile(
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
        record_shapes=config.get("record_shapes", False),
        profile_memory=config.get("profile_memory", True),
        with_stack=config.get("with_stack", True),
    )
    prof.start()
    return prof


def main(argv: Optional[Sequence[str]] = None):
    config = get_config(argv)
    logger = Logger(**config["logger"])

    # Loading train dataset
    logger.info("Load train dataset...")

    train_dataloader_config = config["train_dataloader"]
    train_dataset_cls = get_dataset_cls(train_dataloader_config)
    logger.info(f"Train dataset type: {train_dataset_cls}")

    train_dataset_config = train_dataloader_config["dataset"]
    train_dataset = train_dataset_cls(**train_dataset_config)

    logger.info("Successfully load train dataset")
    train_dataset.describe()
    train_dataloader = DataLoader(
        train_dataset,
        train_dataloader_config["batch_size"],
        shuffle=train_dataloader_config.get("shuffle", False),
        num_workers=train_dataloader_config["num_workers"],
    )

    # Loading val dataset
    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataset_config = val_dataloader_config["dataset"]

    # TODO: Refactor later
    val_dataset_cls = get_dataset_cls(val_dataloader_config)
    logger.info(f"Val dataset type: {val_dataset_cls}")
    train_info_to_val = train_dataset.pop_info()

    val_dataset = val_dataset_cls(**val_dataset_config, **train_info_to_val)
    val_dataset.pop_info()
    val_dataset.describe()

    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=val_dataloader_config["num_workers"],
    )

    logger.info("Successfully load val dataset")

    checkpoint_folder = os.path.dirname(config["checkpoint_path"])
    os.makedirs(checkpoint_folder, exist_ok=True)

    model_config = config["model"]
    model = DeepFM(train_dataset.field_dims, **model_config)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        keys = model.load_state_dict(checkpoint["state_dict"])

        val_metrics = validate_epoch(val_dataloader, model, device)
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    para_dict = {"t_param": [], "default": []}

    for name, p in model.named_parameters():
        if "t_param" in name:
            para_dict["t_param"].append(p)
        else:
            para_dict["default"].append(p)

    optimizer_threshold = torch.optim.SGD(
        [
            {
                "params": para_dict["t_param"],
                "lr": config["opt_embed"]["t_param_lr"],
                "weight_decay": 0,
            },
        ]
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": para_dict["default"],
                "lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
            },
        ]
    )

    optimizers = [optimizer, optimizer_threshold]

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    logger.info(f"Model config: {model_config}")

    train_prof, val_prof = None, None
    if config["enable_profile"]:
        train_prof = init_profiler(config["profilers"]["train_profiler"])
        val_prof = init_profiler(config["profilers"]["val_profiler"])

        config["num_epochs"] = 1

    emb_config = model_config["embedding_config"]
    init_weight_path = config["opt_embed"]["init_weight_path"]
    if "retrain" not in emb_config["name"]:
        os.makedirs(os.path.dirname(init_weight_path), exist_ok=True)
        torch.save(
            {
                "full": model.state_dict(),
                "emb": model.embedding.state_dict(),
            },
            config["opt_embed"]["init_weight_path"],
        )
    else:
        info = torch.load(init_weight_path)
        mask = info["mask"]
        keys = model.load_state_dict(info["full"], False)
        assert len(keys[0]) == 0, f"There are some keys missing: {keys[0]}"
        model.embedding.init_mask(mask_d=mask["mask_d"], mask_e=mask["mask_e"])

    best_auc = 0
    num_epochs = config["num_epochs"]
    try:
        for epoch_idx in range(num_epochs):
            logger.log_metric("Epoch", epoch_idx, epoch_idx)
            train_metrics = train_epoch(
                train_dataloader,
                model,
                optimizers,
                device,
                config["log_step"],
                train_prof,
                alpha=config["opt_embed"]["alpha"],
            )

            train_metrics.update(metrics.get_env_metrics())
            for metric, value in train_metrics.items():
                logger.log_metric(f"train/{metric}", value, epoch_idx)

            if (epoch_idx + 1) % config["validate_step"] == 0:
                val_metrics = validate_epoch(
                    val_dataloader,
                    model,
                    device,
                )

                val_metrics.update(metrics.get_env_metrics())
                sparsity, n_params = model.embedding.get_sparsity(True)

                logger.log_metric("sparsity", sparsity, epoch_idx)
                logger.info(f"{n_params=}")

                for key, value in val_metrics.items():
                    logger.log_metric(f"val/{key}", value, epoch_idx)

                if best_auc < val_metrics["auc"]:
                    logger.info("New best, saving model...")
                    best_auc = val_metrics["auc"]

                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "model_config": model_config,
                        "val_metrics": val_metrics,
                        "field_dims": train_dataset.field_dims,
                    }
                    torch.save(checkpoint, config["checkpoint_path"])

    except KeyboardInterrupt:
        pass

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()


if __name__ == "__main__":
    main()
