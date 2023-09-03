""" Target is to compare multiple hash function under various hyper parameters """
import copy
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import torch
import train_lightgcn
import yaml
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

BASE_CONFIG = "../configs/lightgcn_config.yaml"
BASE_CONFIG = os.path.join(os.path.dirname(__file__), BASE_CONFIG)
HPRAMS_SUPPORT = [bool, str, float, int, None]


def _create_flatten(config, result, cur_key):
    """Recursive call to flatten config dict"""

    for k, v in config.items():
        new_key = f"{cur_key}/{k}"
        if isinstance(v, HPRAMS_SUPPORT):
            result[new_key] = v
        elif isinstance(v, dict):
            _create_flatten(v, result, new_key)
        elif isinstance(v, list):
            for idx, value in enumerate(v):
                _create_flatten(value, result, new_key + idx)
        else:
            logger.debug(f"{cur_key=},{k=},{v=} - Not support {type(v)} to log hparams")

    return result


def create_qr_configs() -> Tuple[List[Dict], List[Dict]]:
    base_dict_emb = dict(name="qr")
    configs = []
    emb_configs = []

    with open(BASE_CONFIG) as fin:
        base_config = yaml.safe_load(fin)

    for divider in [2, 10, 20, 50, 100, None]:
        for ops in ["mult", "cat"]:
            new_emb_config = {
                **base_dict_emb,
                "divider": divider,
                "operation": ops,
            }

            new_config = copy.deepcopy(base_config)

            new_config["model"]["embedding_config"] = new_emb_config
            new_config["logger"]["log_name"] = f"qr_{divider}_{ops}"

            configs.append(new_config)
            emb_configs.append(new_emb_config)

    return configs, emb_configs


def create_dhe_configs() -> Tuple[List[Dict], List[Dict]]:
    base_dict_emb = dict(name="dhe")
    configs = []
    emb_configs = []

    with open(BASE_CONFIG) as fin:
        base_config = yaml.safe_load(fin)

    hidden_size_choices: List[List[int]] = [
        [128],
        [128, 64],
    ]

    for hidden_size in hidden_size_choices:
        for inp_size in [1024, 512, 256]:
            new_emb_config = {
                **base_dict_emb,
                "hidden_sizes": hidden_size,
                "inp_size": inp_size,
            }

            new_config = copy.deepcopy(base_config)

            new_config["model"]["embedding_config"] = new_emb_config

            hidden_size = "-".join(hidden_size)
            new_config["logger"]["log_name"] = f"dhr_{hidden_size}_{inp_size}"

            # Tensorboard cannot store List int hyperparameters
            new_emb_config["hidden_sizes"] = hidden_size

            configs.append(new_config)
            emb_configs.append(new_emb_config)

    return configs, emb_configs


def main():
    qr_configs, emb_configs = create_qr_configs()

    tmp_file = tempfile.mktemp()
    writer = SummaryWriter("logs/summary")

    for qr_config, emb_config in zip(qr_configs, emb_configs):
        with open(tmp_file, "w") as fout:
            yaml.dump(qr_config, fout)

        # Using subprocess to remove torch cuda allocated memory
        subprocess.run(["python", train_lightgcn.__file__, tmp_file])

        torch.cuda.empty_cache()

        checkpoint_path = qr_config["checkpoint_path"]

        metrics = torch.load(checkpoint_path, map_location="cpu")["val_metrics"]

        run_name = qr_config["logger"]["log_name"]
        writer.add_hparams(emb_config, metrics, run_name=run_name)


if __name__ == "__main__":
    main()
