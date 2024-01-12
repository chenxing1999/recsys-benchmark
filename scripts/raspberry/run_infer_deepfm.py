import os
import subprocess
from collections import defaultdict


def get_peak_v0(file_path):
    res = 0
    with open(file_path) as fin:
        for line in fin:
            if line.startswith("MEM"):
                mem = line.rstrip().split()[1]
                res = max(res, float(mem))
    return res


# batchs = [1, 64]
batchs = [64]
checkpoint_dir = "checkpoints/deepfm"
output_dir = "mprof/criteo/infer-original-torchload/"

os.makedirs(output_dir, exist_ok=True)

checkpoints_and_weight_loader = [
    # ("original.pth", "original"),
    # ("original-full.pth", "original"),
    # ("deepfm-qr-5.pth", "qr"),
    # ("cerp-80-no-mask.pth", "cerp"),
    # ("pep-80-csr.pth", "pep"),
    # ("deepfm-dhe-08-v2.pth", "dhe"),
    # ("tt-rec-80-torch.pth", "ttrec-cpu"),
    # uses seperate weight so that the mask
    # will not contributed into RAM memory
    # opt actually could be slightly
    # more optimized (pruning all neuron in the last dimension)
    # ("opt-80-vanilla.pth", "opt-cpu"),
    ("opt-80-vanilla-full.pth", "opt-cpu"),
    # ("opt-80-csr.pth", "opt-sparse-cpu"),
]

n_repeat = 5

out = defaultdict(dict)


# Check if all checkpoint exists
for checkpoint, _ in checkpoints_and_weight_loader:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    assert os.path.exists(checkpoint_path)

for b in batchs:
    for checkpoint, method in checkpoints_and_weight_loader:
        out[method][b] = 0

        # load model
        name = f"{method}-load-model.dat"
        out_path = os.path.join(output_dir, name)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

        cmd = [
            "mprof",
            "run",
            "-o",
            out_path,
            "python",
            "scripts/deepfm/infer_deepfm.py",
            "--train-info",
            "dataset/ctr/criteo/criteo-common-split/train-light.bin",
            "-c",
            checkpoint_path,
            "-m",
            method,
            "-b",
            str(b),
            "-t",
            "load_model",
        ]
        subprocess.run(cmd)
        peak = get_peak_v0(out_path)
        print()
        print(f"{method=} - {checkpoint=} - load model |    {peak=}MiB")
        print()

        for i in range(n_repeat):
            name = f"{method}-batch{b}_v{i}.dat"
            print("currently running", name)
            out_path = os.path.join(output_dir, name)

            cmd = [
                "mprof",
                "run",
                "-o",
                out_path,
                "python",
                "scripts/deepfm/infer_deepfm.py",
                "--train-info",
                "dataset/ctr/criteo/criteo-common-split/train-light.bin",
                "-c",
                checkpoint_path,
                "-m",
                method,
                "-b",
                str(b),
            ]
            subprocess.run(cmd)
            peak = get_peak_v0(out_path)
            out[method][b] += peak

        peak = out[method][b] / n_repeat
        out[method][b] = peak
        print()
        print(f"{method=} - {checkpoint=} - {b=} |    {peak=}MiB")
        print()

        # print("---Get run time---")
        # cmd = [
        #     "python",
        #     "scripts/deepfm/infer_deepfm.py",
        #     "--train-info",
        #     "dataset/ctr/criteo/criteo-common-split/train-light.bin",
        #     "-c",
        #     checkpoint_path,
        #     "-m",
        #     method,
        #     "-b",
        #     str(b),
        # ]
        # subprocess.run(cmd)
