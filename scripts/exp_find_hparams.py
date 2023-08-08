""" Target is to compare multiple hash function under various hyper parameters """
import argparse
import copy
import os
import shutil
import subprocess
import tempfile

import optuna
import torch
import yaml
from loguru import logger

import train_lightgcn

DEFAULT_BASE_CONFIG_PATH = "../configs/lightgcn_config.yaml"
DEFAULT_BASE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), DEFAULT_BASE_CONFIG_PATH
)


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
        "--enable_sgl_wa", default=True, help="Enable SGL-WA backbone or not"
    )
    args = parser.parse_args()

    if args.log_folder is None:
        name = os.path.basename(args.base_config)
        name, _ = os.path.splitext(name)

        args.log_folder = name

    if args.run_name is None:
        name = os.path.basename(args.log_folder)
        args.run_name = name

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
BEST_CHECKPOINT_PATH = "checkpoints/best_checkpoints.pth"


def generate_config(trial):
    """Generate a config yaml from trial"""

    new_config = copy.deepcopy(base_config)

    lr = trial.suggest_float("learning_rate", 5e-4, 1e-2)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
    num_layers = trial.suggest_int("num_layers", 1, 4)

    name = f"lr{lr:.4f}-decay{weight_decay:.4f}-num_layers{num_layers}"
    if ENABLE_SGL_WA:
        info_nce = trial.suggest_float("info_nce_weight", 0, 1, step=0.05)
        new_config["info_nce_weight"] = info_nce
        name += f"-info_nce{info_nce:.4f}"

    new_config["learning_rate"] = lr
    new_config["weight_decay"] = weight_decay
    new_config["model"]["num_layers"] = num_layers

    new_config["logger"]["log_folder"] = LOG_FOLDER
    new_config["logger"]["log_name"] = name
    return new_config


def objective(trial: optuna.Trial):
    new_config = generate_config(trial)

    name = new_config["logger"]["log_name"]
    logger.info(name)

    # Using subprocess to remove torch cuda allocated memory
    tmp_file = tempfile.mktemp()
    with open(tmp_file, "w") as fout:
        yaml.dump(new_config, fout)

    subprocess.run(["python", train_lightgcn.__file__, tmp_file])

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
    study = optuna.create_study(
        "sqlite:///db.sqlite3",
        study_name=RUN_NAME,
        direction="maximize",
        load_if_exists=True,
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

    train_lightgcn.main(["configs/best-trial.yaml"])


if __name__ == "__main__":
    main()
