1. Search for best hyperparameters for NDCG based on target sparsity

(
Note you might need to manual edit the logic in the provided code. The original implementation is designed for target sparsity 80%.
The main thing to edit is the trial pruning logic, which located in `_main` function.
When I implement the original scripts, I use the provided threshold as it usually
provide good baseline to prune bad result.
).

```shell
python scripts/train_lightgcn_pep.py configs/pep_debug_config.yaml
```

2. Retrain the model by edit `config["pep_config"]["is_retrain"]` to True
