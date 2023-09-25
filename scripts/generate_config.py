"""Short script to automatically create new config in retrain mode"""
import argparse
from typing import Optional, Sequence

import yaml


def str2bool(txt: str) -> bool:
    return txt.lower() not in ["false", "0"]


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="Path to input config file")
    parser.add_argument("output_path", help="Path to output config file")

    parser.add_argument(
        "--add-retrain", action="store_true", help="Add retrain to config"
    )
    parser.add_argument(
        "--checkpoint_path", help="Update checkpoint path to new path", default=None
    )
    parser.add_argument(
        "--run_test_mode",
        type=str2bool,
        help="Overide new mode run_test variable",
        default=None,
    )

    return parser.parse_args(argv)


args = parse_args()

with open(args.input_path) as fin:
    config = yaml.safe_load(fin)

if args.add_retrain:
    config["model"]["embedding_config"]["name"] += "_retrain"

    if config["logger"].get("name") is None:
        config["logger"]["name"] = ""
    config["logger"]["name"] += "_retrain"

if args.run_test_mode is not None:
    config["run_test"] = args.run_test_mode

if args.checkpoint_path:
    config["checkpoint_path"] = args.checkpoint_path

with open(args.output_path, "w") as fout:
    yaml.dump(config, fout)
