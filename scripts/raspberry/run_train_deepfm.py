import os
import subprocess
import tempfile
from collections import defaultdict

import yaml


def get_peak_v0(file_path):
    res = 0
    with open(file_path) as fin:
        for line in fin:
            if line.startswith("MEM"):
                mem = line.rstrip().split()[1]
                res = max(res, float(mem))
    return res


def _create_tmp_config(config_path, raspberry_data_config):
    with open(config_path) as fin:
        config = yaml.safe_load(fin)

    for k in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        config[k] = raspberry_data_config[k]
    return config


# batchs = [1, 64]
config_dir = "configs/deepfm"
output_dir = "mprof/criteo/train/"

os.makedirs(output_dir, exist_ok=True)

raspberry_config = "configs/deepfm/base_config_raspberry.yaml"
with open(raspberry_config) as fin:
    raspberry_data_config = yaml.safe_load(fin)

configs_and_weight_loader = [
    # ("base_config.yaml", "original"),
    # ("qr-5.yaml", "qr"),
    # ("cerp_config_retrain-80.yaml", "cerp_retrain"),
    # ("opt_embed_retrain_debug.yaml", "opt_retrain"),
    # ("pep_retrain_sample.yaml", "pep_retrain"),
    ("opt_embed_debug.yaml", "opt"),
    # ("pep_sample.yaml", "pep"),
    # ("cerp_config.yaml", "cerp"),
    # uses seperate weight so that the mask will not contributed into RAM memory
    # opt actually could be slightly
    # more optimized (pruning all neuron in the last dimension)
    # ("opt-80-vanilla.pth", "opt-cpu"),
    # ("opt-80-vanilla-full.pth", "opt-cpu"),
    # ("opt-80-csr.pth", "opt-sparse-cpu"),
]

n_repeat = 5

out = defaultdict(int)


# Check if all checkpoint exists
for config, _ in configs_and_weight_loader:
    config_path = os.path.join(config_dir, config)
    assert os.path.exists(config_path), config_path

for config, method in configs_and_weight_loader:
    out[method] = 0

    config_path = os.path.join(config_dir, config)
    tmp_config = _create_tmp_config(config_path, raspberry_data_config)

    tmp_config_file = tempfile.NamedTemporaryFile(prefix=method)
    tmp_config_file.close()
    with open(tmp_config_file.name, "w") as fout:
        yaml.dump(tmp_config, fout)

    if method == "pep":
        func = "scripts/deepfm/train_deepfm_pep.py"
    elif method == "opt":
        func = "scripts/deepfm/train_deepfm_optembed.py"
    elif method == "cerp":
        func = "scripts/deepfm/train_deepfm_cerp.py"
    else:
        func = "scripts/deepfm/train_deepfm.py"

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
            func,
            tmp_config_file.name,
        ]
        subprocess.run(cmd)
        peak = get_peak_v0(out_path)
        out[method] += peak

    # tmp_config_file.delete()
    os.remove(tmp_config_file.name)
    peak = out[method] / n_repeat
    out[method] = peak
    print()
    print(f"{method=} - {config=} |    {peak=}MiB")
    print()
