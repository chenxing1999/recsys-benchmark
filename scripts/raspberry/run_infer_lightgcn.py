import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import torch
import yaml

SCRIPT_PATH = "scripts/lightgcn/infer_lightgcn.py"
# SCRIPT_PATH = "run.py"


def get_peak_v0(file_path):
    res = 0
    with open(file_path) as fin:
        for line in fin:
            if line.startswith("MEM"):
                mem = line.rstrip().split()[1]
                res = max(res, float(mem))
    return res


def _create_tmp_config(
    config_path,
    checkpoint_path=None,
    num_layers=3,
):
    with open(config_path) as fin:
        config = yaml.safe_load(fin)

    config["model"]["num_layers"] = num_layers
    if checkpoint_path:
        config["checkpoint_path"] = checkpoint_path

    return config


checkpoint_dir = "checkpoints/lightgcn"
config_dir = "configs/yelp2018/"
output_dir = "mprof/yelp2018/infer_v3"

os.makedirs(output_dir, exist_ok=True)


loader_checkpoint_configs = [
    ("original", "sgl-wa.pth", "base_config.yaml"),
    ("qr", "qr-5.pth", "qr-5.yaml"),
    ("dhe", "dhe-best-checkpoints-256-256-128-128.pth", "dhe-256-256-128-128.yaml"),
    ("pep", "pep-best-checkpoints-0.8.pth", "pep_retrain.yaml"),
    ("optemb", "best-opt-80.pth", "opt-embed-retrain.yaml"),
    ("cerp", "yelp2018_cerp-80.pth", "cerp_config-retrain-80.yaml"),
    ("tt_rec_torch", "tt_rec_torch_out.pth", "tt_rec_torch_config.yaml"),
]

loaders, checkpoints, configs = zip(*loader_checkpoint_configs)

assert len(checkpoints) == len(configs)
assert len(checkpoints) == len(loaders)


n_repeat = 30

out = defaultdict(int)


# Check if all checkpoint exists
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    assert os.path.exists(checkpoint_path), checkpoint_path

for config in configs:
    config_path = os.path.join(config_dir, config)
    assert os.path.exists(config_path), config_path

for checkpoint, method, config_name in zip(checkpoints, loaders, configs):
    out[method] = []

    # load model
    name = f"{method}-load-model.dat"
    out_path = os.path.join(output_dir, name)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

    config_path = os.path.join(config_dir, config_name)

    tmp_config = _create_tmp_config(config_path, checkpoint_path)

    tmp_config_file = tempfile.NamedTemporaryFile(prefix=method)
    tmp_config_file.close()
    with open(tmp_config_file.name, "w") as fout:
        yaml.dump(tmp_config, fout)

    tmp_torch_path = "/tmp/torch_model.bin"
    cmd = [
        "python",
        SCRIPT_PATH,
        "-t",
        "4",  # load model and data
        "-m",
        method,
        "-s",
        tmp_torch_path,
        tmp_config_file.name,
    ]
    subprocess.run(cmd)

    tmp_config = _create_tmp_config(
        config_path,
        tmp_torch_path,
    )
    with open(tmp_config_file.name, "w") as fout:
        yaml.dump(tmp_config, fout)

    cmd = [
        "mprof",
        "run",
        "-o",
        out_path,
        "python",
        SCRIPT_PATH,
        "-t",
        "2",  # load model and data
        "-m",
        "torch",
        tmp_config_file.name,
    ]
    print()
    print("Method:", method)
    cmd = [
        "python",
        SCRIPT_PATH,
        "-m",
        "torch",
        tmp_config_file.name,
    ]
    subprocess.run(cmd)
    print()
    continue
    peak = get_peak_v0(out_path)
    print()
    print(f"{method=} - {checkpoint=} - load model |    {peak=}MiB")
    print()

    for i in range(n_repeat):
        name = f"{method}_v{i}.dat"
        print("currently running", name)
        out_path = os.path.join(output_dir, name)

        cmd = [
            "mprof",
            "run",
            "-o",
            out_path,
            "python",
            SCRIPT_PATH,
            "-t",
            "0",  # load model and data
            "-m",
            "torch",
            "-n",
            "1",
            tmp_config_file.name,
        ]
        subprocess.run(cmd)
        peak = get_peak_v0(out_path)
        out[method].append(peak)

    os.remove(tmp_config_file.name)

    np_arr = np.array(out[method])
    mean = np_arr.mean()
    max_mem = np_arr.max()
    min_mem = np_arr.min()
    std_mem = np_arr.std()

    final_out = os.path.join(output_dir, f"all_{method}.bin")
    torch.save(out[method], final_out)

    print()
    print(
        f"{method=} - {checkpoint=} \n"
        f"peak={mean}+-{std_mem}MiB \n"
        f"{min_mem=} | {max_mem=}"
    )
    print()

torch.save(output_dir, "all.bin")
