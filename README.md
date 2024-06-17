# RecSys Benchmark

## Installation

```shell
# clone this repo
git clone git@github.com:chenxing1999/recsys-benchmark.git

# cd to folder
cd recsys-benchmark

# Create virtual env
python -m venv env
source env/bin/activate

# install
pip install -e '.[dev]'
```

## Quickstart

Train SGL-WA:

```shell
python scripts/lightgcn/train_lightgcn.py configs/yelp2018/base_config.yaml
```

Run Hyperparam search with Optuna

```shell
mkdir -p checkpoints/lightgcn/sgl-wa.pth
python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/base_config.yaml -l logs/sgl-wa -p checkpoints/lightgcn/sgl-wa.pth
```

For more detail with LightGCN, click [here](./docs/lightgcn_tutorial.md)

For more detail with DeepFM, click [here](./docs/deepfm_tutorial.md). DCN-Mix uses the same API with DeepFM.

## Other artifacts

TBD

## Acknowledge

This source code is based on:

```
https://github.com/gusye1234/LightGCN-PyTorch
https://github.com/rixwew/pytorch-fm
```
