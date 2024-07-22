""" Target is to compare multiple hash function under various hyper parameters """
import argparse
import copy
import os
import shutil
import subprocess
import tempfile

import optuna
import torch
import train_deepfm
import yaml
from loguru import logger

DEFAULT_BASE_CONFIG_PATH = "../configs/deepfm/base_config.yaml"
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
        "--best_checkpoint_path",
        "-p",
        default=DEFAULT_BEST_CHECKPOINT_PATH,
        help="Path to best checkpoint path",
    )
    parser.add_argument(
        "--tmp_checkpoint_path",
        default=None,
        help="Path to tmp trial checkpoint path",
    )
    parser.add_argument(
        "--n_trials",
        default=30,
        type=int,
        help="Number of trials for Optuna",
    )
    parser.add_argument(
        "--db",
        default="sqlite:///db-deepfm.sqlite3",
        type=str,
        help="Path to Optuna Database",
    )
    parser.add_argument(
        "--disable_subprocess",
        action="store_true",
        help="Disable subprocess version. In short, enable subprocess"
        "-> cannot edit code, same model init every run.",
    )
    parser.add_argument(
        "--max_trials",
        default=None,
        type=int,
        help="Maximum trials. Set this for multiple parallel runs",
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

    return args


args = parse_args()
# input variable
BASE_CONFIG = args.base_config
LOG_FOLDER = args.log_folder

RUN_NAME = args.run_name


with open(BASE_CONFIG) as fin:
    base_config = yaml.safe_load(fin)

if args.tmp_checkpoint_path:
    base_config["checkpoint_path"] = args.tmp_checkpoint_path
CHECKPOINT_PATH = base_config["checkpoint_path"]
BEST_CHECKPOINT_PATH = args.best_checkpoint_path


def generate_config(trial):
    """Generate a config yaml from trial"""

    new_config = copy.deepcopy(base_config)

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0, 1)

    name = f"lr{lr:.4f}-decay{weight_decay:.4f}-dropout{dropout:.4f}"
    new_config["learning_rate"] = lr
    new_config["weight_decay"] = weight_decay
    new_config["model"]["p_dropout"] = dropout

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

    if args.disable_subprocess:
        train_deepfm.main([tmp_file])
    else:
        subprocess.run(["python", train_deepfm.__file__, tmp_file])

    torch.cuda.empty_cache()

    checkpoint_path = new_config["checkpoint_path"]

    metrics = torch.load(checkpoint_path, map_location="cpu")["val_metrics"]
    return metrics["auc"]


def save_best_on_val_callbacks(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        if previous_best_value is None:
            previous_best_value = 0
        logger.info(
            f"Save best. Cur best={study.best_value:.4f}."
            f"Prev best={previous_best_value:.4f}."
        )
        study.set_user_attr("previous_best_value", study.best_value)
        shutil.copy(CHECKPOINT_PATH, BEST_CHECKPOINT_PATH)

        with open("configs/best-trial.yaml", "w") as fout:
            yaml.dump(generate_config(frozen_trial), fout)


def main():
    sampler = optuna.samplers.TPESampler(
        seed=2023
    )  # Make the sampler behave in a deterministic way.

    callbacks = [save_best_on_val_callbacks]
    if args.max_trials:
        callbacks.append(optuna.study.MaxTrialsCallback(args.max_trials))

    study = optuna.create_study(
        args.db,
        study_name=RUN_NAME,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(
        objective,
        args.n_trials,
        callbacks=callbacks,
    )
    trial = study.best_trial

    logger.info(trial.params)
    best_config = generate_config(trial)

    # Get test accuracy
    best_config["run_test"] = True
    best_config["checkpoint_path"] = BEST_CHECKPOINT_PATH
    with open("configs/best-trial.yaml", "w") as fout:
        yaml.dump(best_config, fout)

    train_deepfm.main(["configs/best-trial.yaml"])


if __name__ == "__main__":
    main()
