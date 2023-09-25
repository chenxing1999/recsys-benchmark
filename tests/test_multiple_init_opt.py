import copy
import os
import subprocess
import tempfile

import torch
import yaml

TEST_FOLDER = os.path.dirname(__file__)
BASE_CONFIG_PATH = os.path.join(TEST_FOLDER, "assets/lightgcn_config.yaml")
with open(BASE_CONFIG_PATH) as fin:
    BASE_CONFIG = yaml.safe_load(fin)


TRAIN_SCRIPT = os.path.join(TEST_FOLDER, "../scripts/lightgcn/train_lightgcn.py")
HIDDEN_SIZE = 64
NUM_USER = 101
NUM_ITEM = 77


def test_multiple_init_for_opt():
    config = copy.deepcopy(BASE_CONFIG)
    config["num_epochs"] = 0

    with tempfile.TemporaryDirectory("wr") as tmpdir:
        init_weight_path = os.path.join(tmpdir, "initial.pth")
        config["opt_embed"] = {
            "init_weight_path": init_weight_path,
            "t_param_lr": 0.0001,
            "alpha": 0.0001,
        }
        config["model"]["embedding_config"]["name"] = "optembed_d"

        tmp_config = os.path.join(tmpdir, "config.yaml")

        with open(tmp_config, "w") as fout:
            yaml.dump(config, fout)

        subprocess.run(["python", TRAIN_SCRIPT, tmp_config])

        checkpoint = torch.load(init_weight_path)
        checkpoint["mask"] = {"user": {}, "item": {}}
        checkpoint["mask"]["user"]["mask_d"] = torch.randint(
            HIDDEN_SIZE, size=(NUM_USER,)
        )
        checkpoint["mask"]["item"]["mask_d"] = torch.randint(
            HIDDEN_SIZE, size=(NUM_ITEM,)
        )
        torch.save(checkpoint, init_weight_path)

        config["model"]["embedding_config"]["name"] = "optembed_d_retrain"

        subprocess.run(["python", TRAIN_SCRIPT, tmp_config])
        checkpoint2 = torch.load(init_weight_path)

        assert len(checkpoint["full"]) == len(checkpoint2["full"])
        for key in checkpoint["full"]:
            assert (checkpoint["full"][key] == checkpoint2["full"][key]).all()
