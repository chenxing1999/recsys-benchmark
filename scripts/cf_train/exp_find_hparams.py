""" Code to find best hyperparameter for NeuralCF (MLP) """
import argparse
import copy
import os
import shutil
import subprocess
import tempfile

import optuna
import torch
import train_cf
import yaml
from loguru import logger

from src.utils import set_seed

set_seed(2023)

DEFAULT_BASE_CONFIG_PATH = "../configs/lightgcn_config.yaml"
DEFAULT_BASE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), DEFAULT_BASE_CONFIG_PATH
)
DEFAULT_BEST_CHECKPOINT_PATH = "checkpoints/best_checkpoints.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--base_config", default=DEFAULT_BASE_CONFIG_PATH)
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
        "--disable_sgl_wa",
        action="store_true",
        help="Disable SGL-WA backbone or not. Default: False",
    )
    parser.add_argument(
        "--best_checkpoint_path",
        "-p",
        default=DEFAULT_BEST_CHECKPOINT_PATH,
        help="Path to best checkpoint path",
    )
    args = parser.parse_args()

    logger.debug(args)
    if args.log_folder is None:
        name = os.path.basename(args.base_config)
        name, _ = os.path.splitext(name)

        args.log_folder = name

    if args.run_name is None:
        log_folder = args.log_folder
        if log_folder.endswith("/"):
            log_folder = log_folder.rstrip("/")

        name = os.path.basename(log_folder)
        args.run_name = name

    # making checkpoint folder if it not exists
    checkpoint_folder = os.path.dirname(args.best_checkpoint_path)
    os.makedirs(checkpoint_folder, exist_ok=True)

    hparams_log_path = os.path.join(args.log_folder, "log")
    logger.add(hparams_log_path)
    setattr(args, "enable_sgl_wa", not args.disable_sgl_wa)

    return args


args = parse_args()
# input variable
BASE_CONFIG = args.base_config
LOG_FOLDER = args.log_folder
ENABLE_SGL_WA = args.enable_sgl_wa

RUN_NAME = args.run_name


with open(BASE_CONFIG) as fin:
    base_config = yaml.safe_load(fin)

CHECKPOINT_PATH = base_config["checkpoint_path"]
BEST_CHECKPOINT_PATH = args.best_checkpoint_path


def generate_config(trial):
    """Generate a config yaml from trial"""

    new_config = copy.deepcopy(base_config)

    # It is better practice to keep log=True
    # (sampling float from log uniform distribution). However, emperical
    # results show that log=False provide better result on base model
    lr = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=False)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=False)
    dropout = trial.suggest_float("dropout", 0, 1)
    num_neg = trial.suggest_int("num_neg", 1, 5)

    name = f"lr{lr:.4f}-dropout{dropout:.4f}-num_neg{num_neg}-decay{weight_decay:.4f}"

    new_config["learning_rate"] = lr
    new_config["weight_decay"] = weight_decay
    new_config["model"]["p_dropout"] = dropout

    new_config["train_dataloader"]["dataset"]["num_neg_item"] = num_neg

    new_config["logger"]["log_folder"] = LOG_FOLDER
    new_config["logger"]["log_name"] = name
    return new_config


def objective(trial: optuna.Trial):
    new_config = generate_config(trial)

    name = new_config["logger"]["log_name"]
    logger.info(name)

    tmp_file = tempfile.mktemp()
    with open(tmp_file, "w") as fout:
        yaml.dump(new_config, fout)

    # use subprocess to keep seed constant
    subprocess.run(["python", train_cf.__file__, tmp_file])

    torch.cuda.empty_cache()

    checkpoint_path = new_config["checkpoint_path"]

    metrics = torch.load(checkpoint_path, map_location="cpu")["val_metrics"]
    return metrics["ndcg"]


def save_best_on_val_callbacks(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        shutil.copy(CHECKPOINT_PATH, BEST_CHECKPOINT_PATH)

        with open("configs/best-trial.yaml", "w") as fout:
            yaml.dump(generate_config(frozen_trial), fout)


def main():
    sampler = optuna.samplers.TPESampler(
        seed=2023
    )  # Make the sampler behave in a deterministic way.

    study = optuna.create_study(
        "sqlite:///db.sqlite3",
        study_name=RUN_NAME,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(objective, 30, callbacks=[save_best_on_val_callbacks])
    trial = study.best_trial

    logger.info(trial.params)
    best_config = generate_config(trial)

    # Get test accuracy
    best_config["run_test"] = True
    best_config["checkpoint_path"] = BEST_CHECKPOINT_PATH
    with open("configs/best-trial.yaml", "w") as fout:
        yaml.dump(best_config, fout)

    train_cf.main(["configs/best-trial.yaml"])


if __name__ == "__main__":
    main()
