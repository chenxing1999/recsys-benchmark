"""Script to train LightGCN with PEP config"""
import argparse
import copy
import os
import shutil
from functools import partial
from typing import Dict, Optional, Sequence, Union

import loguru
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.models import LightGCN, get_graph_model
from src.models.embeddings import PepEmbeeding, RetrainPepEmbedding
from src.trainer.lightgcn import train_epoch, validate_epoch

IPepEmbedding = Union[PepEmbeeding, RetrainPepEmbedding]
N_TRIALS = 30


def get_config(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)
    return config


def generate_config(trial, base_config, enable_sgl_wa=True):
    """Generate a config yaml from trial"""

    new_config = copy.deepcopy(base_config)

    lr = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    init_threshold = trial.suggest_float("init_threshold", -10, -5, step=0.1)

    name = f"lr{lr:.4f}-decay{weight_decay:.4f}-num_layers{num_layers}"
    if enable_sgl_wa:
        info_nce = trial.suggest_float("info_nce_weight", 0, 1, step=0.05)
        new_config["info_nce_weight"] = info_nce
        name += f"-info_nce{info_nce:.4f}"

    new_config["learning_rate"] = lr
    new_config["weight_decay"] = weight_decay
    new_config["model"]["num_layers"] = num_layers

    pep_config = base_config["pep_config"]
    if not pep_config["is_retrain"]:
        new_config["model"]["embedding_config"]["init_threshold"] = init_threshold

    # new_config["logger"]["log_folder"] = LOG_FOLDER
    new_config["logger"]["log_name"] = name
    return new_config


def _validate_config(config):
    assert config["model"]["embedding_config"]["name"].startswith("pep")
    assert config["model"]["name"] == "lightgcn"

    assert "pep_config" in config


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


def _main(trial: optuna.Trial, base_config: Dict):
    config = generate_config(trial, base_config)
    logger = Logger(**config["logger"])

    # Not support other config besides pep config
    config = generate_config(trial, config)

    # Loading train dataset
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

    if config["run_test"]:
        val_dataloader_config = config["test_dataloader"]
    else:
        val_dataloader_config = config["val_dataloader"]

    logger.info("Load val dataset...")
    val_dataset = TestCFGraphDataset(val_dataloader_config["dataset"]["path"])
    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        collate_fn=TestCFGraphDataset.collate_fn,
        num_workers=val_dataloader_config["num_workers"],
    )
    logger.info("Successfully load val dataset")

    checkpoint_folder = os.path.dirname(config["checkpoint_path"])
    os.makedirs(checkpoint_folder, exist_ok=True)

    model_config = config["model"]
    model: LightGCN = get_graph_model(
        train_dataset.num_users,
        train_dataset.num_items,
        model_config,
    )

    model.user_emb_table: IPepEmbedding
    model.item_emb_table: IPepEmbedding

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"])

        val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    logger.info(f"Model config: {model_config}")
    is_pep_retrain = model_config["embedding_config"]["name"] == "pep_retrain"

    train_prof, val_prof = None, None
    if config["enable_profile"]:
        train_prof = init_profiler(config["profilers"]["train_profiler"])
        val_prof = init_profiler(config["profilers"]["val_profiler"])

        config["num_epochs"] = 1

    best_ndcg = 0
    achieved_target = False
    num_epochs = config["num_epochs"]

    # Saving and revert
    pep_config = config["pep_config"]
    model_init_path = pep_config["model_init_path"]
    target_sparsity = pep_config["target_sparsity"]
    trial_checkpoint = pep_config["trial_checkpoint"]
    os.makedirs(os.path.join(trial_checkpoint, "user"), exist_ok=True)
    os.makedirs(os.path.join(trial_checkpoint, "item"), exist_ok=True)
    if os.path.exists(model_init_path):
        state = torch.load(model_init_path)
        model.load_state_dict(state, strict=False)
    else:
        dirname = os.path.dirname(model_init_path)
        os.makedirs(dirname, exist_ok=True)
        state = model.state_dict()
        torch.save(state, model_init_path)
    # end hacking

    num_users = train_dataset.num_users
    num_items = train_dataset.num_items
    for epoch_idx in range(num_epochs):
        logger.log_metric("Epoch", epoch_idx, epoch_idx)
        train_metrics = train_epoch(
            train_dataloader,
            model,
            optimizer,
            device,
            config["log_step"],
            config["weight_decay"],
            train_prof,
            info_nce_weight=config["info_nce_weight"],
        )

        train_metrics.update(metrics.get_env_metrics())
        for metric, value in train_metrics.items():
            logger.log_metric(f"train/{metric}", value, epoch_idx)

        if (epoch_idx + 1) % config["validate_step"] == 0:
            val_metrics = validate_epoch(
                train_dataset,
                val_dataloader,
                model,
                device,
                filter_item_on_train=True,
                profiler=val_prof,
            )

            val_metrics.update(metrics.get_env_metrics())

            for key, value in val_metrics.items():
                logger.log_metric(f"val/{key}", value, epoch_idx)

            if best_ndcg < val_metrics["ndcg"]:
                logger.info("New best, saving model...")
                best_ndcg = val_metrics["ndcg"]

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, config["checkpoint_path"])

            if not is_pep_retrain:
                # get sparsity and store weight if possible
                user_sparsity = model.user_emb_table.get_sparsity()
                logger.log_metric("sparsity/user", user_sparsity, epoch_idx)

                item_sparsity = model.item_emb_table.get_sparsity()
                logger.log_metric("sparsity/item", item_sparsity, epoch_idx)

                sparsity = (num_users * user_sparsity + num_items * item_sparsity) / (
                    num_users + num_items
                )
                if epoch_idx == 1 and sparsity < 0.01:
                    raise optuna.TrialPruned()
                elif epoch_idx == 5:
                    if sparsity < 0.4 or best_ndcg < 0.002:
                        raise optuna.TrialPruned()

                # Check if sparsity is low enough and save result mask
                model.item_emb_table.train_callback()
                model.user_emb_table.train_callback()
                if not achieved_target and sparsity > target_sparsity:
                    logger.info("found sparsity target")
                    achieved_target = True
                    user_path = os.path.join(trial_checkpoint, "user/target.pth")
                    torch.save(model.user_emb_table.state_dict(), user_path)
                    item_path = os.path.join(trial_checkpoint, "item/target.pth")
                    torch.save(model.item_emb_table.state_dict(), item_path)

    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()

    # get best metric
    user_sparsity = model.user_emb_table.get_sparsity()
    item_sparsity = model.item_emb_table.get_sparsity()

    sparsity = (num_users * user_sparsity + num_items * item_sparsity) / (
        num_users + num_items
    )
    return checkpoint["val_metrics"]["ndcg"], sparsity


class Callback(object):
    def __init__(self, config: Dict):
        self.config = config

        pep_config = config["pep_config"]
        self.target_sparsity = pep_config["target_sparsity"]
        self._cur_best_metric = 0
        self._best_checkpoint_path = pep_config["best_checkpoint_dir"]
        self._save_checkpoint_path = pep_config["trial_checkpoint"]

    def __call__(self, study, frozen_trial):
        if frozen_trial.state == optuna.trial.TrialState.PRUNED:
            return
        ndcg, sparsity = frozen_trial.values
        if sparsity >= self.target_sparsity and ndcg > self._cur_best_metric:
            self._cur_best_metric = ndcg
            if os.path.exists(self._best_checkpoint_path):
                shutil.rmtree(self._best_checkpoint_path)
            shutil.copytree(
                self._save_checkpoint_path,
                self._best_checkpoint_path,
            )


def main(argv=None):
    base_config = get_config(argv)
    objective = partial(lambda trial: _main(trial, base_config))
    callback = Callback(base_config)
    study = optuna.create_study(
        "sqlite:///db-pep.sqlite3",
        study_name="pep",
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )

    _validate_config(base_config)
    is_retrain = base_config["pep_config"]["is_retrain"]

    logger = loguru.logger
    if not is_retrain:
        study.enqueue_trial(
            {
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
                "init_threshold": -7,
                "num_layers": 3,
                "info_nce_weight": 0.1,
            }
        )

        study.optimize(objective, N_TRIALS, callbacks=[callback])

        logger.info("Finished searching for best Pep config")
    else:
        pep_config = base_config["pep_config"]

        logger.info("Start retraining best PEP config")
        # Retrain best model logic
        target_sparsity = pep_config["target_sparsity"]

        # Find result with highest ndcg that are s.t. sparsity >= target_sparsity
        best_trial = None
        best_metric = 0
        for trial in study.get_trials():
            score, sparsity = trial.values
            if sparsity >= target_sparsity and score > best_metric:
                best_trial = trial
                best_metric = score

        logger.info(f"Best trial={best_trial}")

        logger.info("Overriding retrain config to base config")

        embedding_config = {}
        embedding_config["name"] = "pep_retrain"
        embedding_config["checkpoint_weight_dir"] = pep_config["best_checkpoint_dir"]
        embedding_config["sparsity"] = "target"

        base_config["model"]["embedding_config"] = embedding_config

        ndcg, sparsity = _main(best_trial, base_config)
        logger.info(f"Finished retrain logic. {sparsity=:.4f} - {ndcg=:.4f}")


if __name__ == "__main__":
    main()
