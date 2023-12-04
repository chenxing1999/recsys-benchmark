Tutorial for training embedding compression with multiple steps for LightGCN.
Note: All of this example code are intended to be performed at root folder of this repo.

# PEP

1. Pruning step

```shell
# use special script for pruning lightgcn
python scripts/lightgcn/train_lightgcn_pep.py -l logs/pep-pruning --n-trials 0 configs/yelp2018/pep_debug_config.yaml
```

2. Retrain step

```shell
python scripts/lightgcn/train_lightgcn.py configs/yelp2018/pep_retrain.yaml
```

# OptEmbed

1. Train supernet

- Use only mask D

```shell
python scripts/lightgcn/train_lightgcn.py configs/yelp2018/opt-embed.yaml
```

- Use mask E with mask D:

```shell
python scripts/lightgcn/train_lightgcn_optembed.py configs/yelp2018/opt-embed-full.yaml
```

2. Run evolutionary

- Uniform distribution (Original OptEmbed paper)

```shell
python scripts/lightgcn/run_opt_evol_lightgcn.py configs/yelp2018/opt-embed.yaml
```

- Modified distribution to get target sparsity

```shell
python scripts/lightgcn/run_opt_evol_lightgcn.py configs/yelp2018/opt-embed.yaml --target-sparsity 0.8
```

3. Run retrain

```shell
python scripts/lightgcn/train_lightgcn.py configs/yelp2018/opt-embed-retrain.yaml
```

# CERP

- Run find mask

```shell
python scripts/lightgcn/train_lightgcn_cerp.py --n-trials 0 -l logs/cerp-80 configs/yelp2018/cerp_config-80.yaml
```

- Run retrain mask

```shell
python scripts/lightgcn/train_lightgcn.py configs/yelp2018/cerp_config-retrain-80.yaml
```
