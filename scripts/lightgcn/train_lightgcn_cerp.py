import argparse
import copy
import os
import shutil
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import loguru
import optuna
import torch
import yaml
from loguru import logger as loguru_logger
from torch.utils.data import DataLoader

from src import metrics
from src.dataset.cf_graph_dataset import CFGraphDataset, TestCFGraphDataset
from src.loggers import Logger
from src.models import get_graph_model, save_cf_emb_checkpoint
from src.models.embeddings.cerp_embedding_utils import train_epoch_cerp
from src.trainer.lightgcn import validate_epoch
from src.utils import set_seed

set_seed(2023)


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[Dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "--log_folder",
        "-l",
        default=None,
        help="Folder to save all of logs, default: filename of config without ext",
    )
    parser.add_argument(
        "--run_name",
        "-n",
        default=None,
        help="Run name, default: basename of log_folder",
    )
    parser.add_argument(
        "--use-tpe",
        action="store_true",
        help="Optimize only ndcg",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Num trials to run. 0 to use base config",
    )

    args = parser.parse_args(argv)
    loguru_logger.debug(args)
    if args.log_folder is None:
        name = os.path.basename(args.config_file)
        name, _ = os.path.splitext(name)

        args.log_folder = name

    if args.run_name is None:
        log_folder = args.log_folder
        if log_folder.endswith("/"):
            log_folder = log_folder.rstrip("/")

        name = os.path.basename(log_folder)
        args.run_name = name

    hparams_log_path = os.path.join(args.log_folder, "log")
    loguru_logger.add(hparams_log_path)

    with open(args.config_file) as fin:
        config = yaml.safe_load(fin)

    config["logger"]["log_folder"] = args.log_folder
    if args.n_trials == 0:
        config["force"] = True
    else:
        config["force"] = False

    return config, args


def generate_config(trial, base_config, enable_sgl_wa=True):
    """Generate a config yaml from trial"""

    new_config = copy.deepcopy(base_config)
    if base_config["force"]:
        return new_config

    lr = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=False)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=False)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    init_threshold = trial.suggest_float("init_threshold", -50, -5, step=0.1)

    name = f"lr{lr:.4f}-decay{weight_decay:.4f}-layers{num_layers}"
    if enable_sgl_wa:
        info_nce = trial.suggest_float("info_nce_weight", 0, 1, step=0.05)
        new_config["info_nce_weight"] = info_nce
        name += f"-info_nce{info_nce:.4f}"

    new_config["learning_rate"] = lr
    new_config["weight_decay"] = weight_decay
    new_config["model"]["num_layers"] = num_layers

    cerp_config = base_config["cerp"]
    if not cerp_config["is_retrain"]:
        new_config["model"]["embedding_config"]["threshold_init"] = init_threshold
        cerp_weight_decay = trial.suggest_float("cerp_weight_decay", 1e-5, 1, log=True)

        new_config["cerp"]["weight_decay"] = cerp_weight_decay
        name += f"threshold{init_threshold:.4f}-t_decay{cerp_weight_decay:.4f}"

    # new_config["logger"]["log_folder"] = LOG_FOLDER
    new_config["logger"]["log_name"] = name
    return new_config


def _validate_config(config):
    assert config["model"]["embedding_config"]["name"].startswith("cerp")

    assert "cerp" in config


def _main(trial, base_config):
    config = generate_config(trial, base_config)
    loguru.logger.info(trial.params)
    logger = Logger(**config["logger"])

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
    model = get_graph_model(
        train_dataset.num_users,
        train_dataset.num_items,
        model_config,
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config["run_test"]:
        checkpoint = torch.load(config["checkpoint_path"])
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        val_metrics = validate_epoch(train_dataset, val_dataloader, model, device)
        for key, value in val_metrics.items():
            logger.info(f"{key} - {value:.4f}")

        return

    weight_decay_threshold = config["cerp"]["weight_decay"]
    weight_decay_model = config["cerp"].get(
        "model_weight_decay", weight_decay_threshold
    )
    params = [
        # threshold
        {"weight_decay": weight_decay_threshold, "params": []},
        # normal weight
        # {"weight_decay": 0, "params": []},
        {"weight_decay": weight_decay_model, "params": []},
    ]
    for name, p in model.named_parameters():
        if "threshold" in name:
            params[0]["params"].append(p)
        else:
            params[1]["params"].append(p)

    optimizer = torch.optim.Adam(
        params,
        lr=config["learning_rate"],
    )

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    logger.log_metric("num_params", num_params)

    logger.info(f"Model config: {model_config}")

    train_prof, val_prof = None, None
    best_ndcg = 0
    num_epochs = config["num_epochs"]

    early_stop_count = 0
    early_stop_config = config.get("early_stop_patience", 0)
    warmup = config.get("warmup", 0)

    cerp_config = config.get("cerp", {"gamma_init": 1.0, "gamma_decay": 0.5})
    gamma_init = cerp_config["gamma_init"]
    gamma_decay = cerp_config["gamma_decay"]
    target_sparsity = cerp_config["target_sparsity"]
    trial_checkpoint = cerp_config["trial_checkpoint"]

    logger.info("Save initial checkpoint")
    save_cf_emb_checkpoint(model, trial_checkpoint, "initial")

    for epoch_idx in range(num_epochs):
        logger.log_metric("Epoch", epoch_idx, epoch_idx)
        train_metrics = train_epoch_cerp(
            train_dataloader,
            model,
            optimizer,
            device,
            config["log_step"],
            config["weight_decay"],
            train_prof,
            info_nce_weight=config["info_nce_weight"],
            prune_loss_weight=(gamma_decay**epoch_idx) * gamma_init,
            target_sparsity=target_sparsity,
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
                    "num_users": train_dataset.num_users,
                    "num_items": train_dataset.num_items,
                }
                torch.save(checkpoint, config["checkpoint_path"])
                early_stop_count = 0
            elif warmup <= epoch_idx:
                early_stop_count += 1
                logger.debug(f"{early_stop_count=}")

                if early_stop_config and early_stop_count > early_stop_config:
                    break

        if train_metrics["sparsity"] >= target_sparsity:
            logger.info("Found target sparsity")
            save_cf_emb_checkpoint(model, trial_checkpoint)
            break

    trial.set_user_attr("diff_sparsity", target_sparsity - train_metrics["sparsity"])
    trial.set_user_attr("sparsity", train_metrics["sparsity"])
    if config["enable_profile"]:
        train_prof.stop()
        val_prof.stop()

    return best_ndcg


def constraint(trial: optuna.Trial) -> Sequence[float]:
    """Constraint function in Optuna format, value > 0 means constraint is violated"""
    return [trial.user_attrs["diff_sparsity"]]


class Callback(object):
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args

        cerp_config = config["cerp"]
        self.target_sparsity = cerp_config["target_sparsity"]
        self._cur_best_metric = 0
        self._best_checkpoint_path = cerp_config["best_checkpoint_dir"]
        self._save_checkpoint_path = cerp_config["trial_checkpoint"]

    def __call__(self, study, frozen_trial):
        if frozen_trial.state == optuna.trial.TrialState.PRUNED:
            return
        if not self.args.use_tpe:
            ndcg, sparsity = frozen_trial.values
        else:
            ndcg = frozen_trial.value
            sparsity = frozen_trial.user_attrs["sparsity"]

        if sparsity >= self.target_sparsity and ndcg > self._cur_best_metric:
            self._cur_best_metric = ndcg
            if os.path.exists(self._best_checkpoint_path):
                shutil.rmtree(self._best_checkpoint_path)
            shutil.copytree(
                self._save_checkpoint_path,
                self._best_checkpoint_path,
            )


def _create_grid_sampler():
    search_space = {
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "cerp_weight_decay": [1e-5, 1e-6, 1e-7],
        "weight_decay": [0],
        "num_layers": [4],
        "init_threshold": [-100],
        "info_nce_weight": [0],
    }
    return optuna.samplers.GridSampler(search_space, seed=2023)


def main(argv=None):
    base_config, args = parse_args(argv)

    callback = Callback(base_config, args)
    if args.n_trials == 0:
        _main(optuna.create_trial(value=0), base_config)
        return
    if args.use_tpe:
        objective = partial(_main, base_config=base_config)
        sampler = optuna.samplers.TPESampler(
            seed=2023,
            constraints_func=constraint,
        )  # Make the sampler behave in a deterministic way.
        # sampler = _create_grid_sampler()
        kwargs = {"direction": "maximize"}
    else:
        sampler = optuna.samplers.NSGAIISampler(
            seed=2023,
            constraints_func=constraint,
        )  # Make the sampler behave in a deterministic way.
        objective = partial(_main, base_config=base_config)
        kwargs = {"directions": ["maximize", "maximize"]}

    study = optuna.create_study(
        "sqlite:///db-cerp.sqlite3",
        study_name=args.run_name,
        load_if_exists=True,
        sampler=sampler,
        **kwargs,
    )

    _validate_config(base_config)
    is_retrain = base_config["cerp"]["is_retrain"]

    logger = loguru.logger
    if not is_retrain:
        study.optimize(objective, args.n_trials, callbacks=[callback])

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
