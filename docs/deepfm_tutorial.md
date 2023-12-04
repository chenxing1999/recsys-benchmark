Tutorial for training embedding compression with multiple steps for DeepFM.
Note: All of this example code are intended to be performed at root folder of this repo.

# PEP

1. Pruning step

```shell
python scripts/deepfm/train_deepfm_pep.py -l logs/pep-pruning configs/deepfm/pep_debug.yaml
```

2. Retrain step

```shell
python scripts/deepfm/train_deepfm.py configs/deepfm/pep_retrain_sample.yaml
```

# OptEmbed

1. Train supernet

- Use only mask D
  TBD

- Use mask E with mask D:

```shell
python scripts/deepfm/train_deepfm_optembed.py configs/deepfm/opt_embed_debug.yaml
```

2. Run evolutionary

- Uniform distribution (Original OptEmbed paper)

```shell
python scripts/deepfm/run_opt_evol.py configs/deepfm/opt_embed_debug.yaml
```

- ~Modified distribution to get target sparsity~ Force minimum sparsity. (For DeepFM I always use uniform distribution as its already possible to provide required distribution).

```shell
python scripts/deepfm/run_opt_evol.py configs/deepfm/opt_embed_debug.yaml
```

3. Run retrain

```shell
python scripts/deepfm/train_deepfm.py configs/deepfm/opt_embed_retrain_debug.yaml
```

# CERP

- Run find mask

```shell
python scripts/deepfm/train_deepfm_cerp.py configs/deepfm/cerp_config.yaml
```

- Run retrain mask

```shell
python scripts/deepfm/train_deepfm.py configs/deepfm/cerp_config-retrain-80.yaml
```
