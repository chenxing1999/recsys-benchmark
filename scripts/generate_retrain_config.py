"""Short script to automatically create new config in retrain mode"""
import sys

import yaml

assert len(sys.argv) == 3, "There is not enough inputs"
with open(sys.argv[1]) as fin:
    config = yaml.safe_load(fin)

config["model"]["embedding_config"]["name"] += "_retrain"
with open(sys.argv[2]) as fout:
    yaml.dump(config, fout)
