import argparse
from typing import Optional, Sequence

import torch
from sklearn.metrics import roc_auc_score

from src.dataset.criteo.utils import preprocess
from src.models.deepfm import DeepFM
from src.utils import set_seed

set_seed(2023)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        "-c",
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--train-info",
        help="Path to train info, used to load feat_mappers and defaults",
    )

    parser.add_argument("--batch_size", "-b", default=64)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    # Load checkpoint
    model = DeepFM.load(args.checkpoint_path)

    # preprocess data
    inps = []
    labels = []
    with open("tests/assets/train_criteo_sample.txt") as fin:
        for idx, line in enumerate(fin):
            inp = fin.readline()
            labels.append(int(inp[0]))

            train_info = torch.load(args.train_info)
            inp_tensor = preprocess(train_info, inp).unsqueeze(0)
            inps.append(inp_tensor)
            if idx == args.batch_size:
                break

    inps = torch.cat(inps, dim=0)
    # infer
    model.eval()
    with torch.no_grad():
        out = model(inps).cpu()

    # print(torch.sigmoid(out).to(torch.int32).tolist());
    # print(labels)
    print(roc_auc_score(torch.tensor(labels), out))


if __name__ == "__main__":
    main()
