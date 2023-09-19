mkdir -p checkpoints/lightgcn/

# SGL-WA base config
python scripts/lightgcn/exp_find_hparams.py -c configs/lightgcn/base_config.yaml \
    -l logs/sgl-wa-base/ \
    -p checkpoints/lightgcn/sgl-wa.pth

# QR----------------------------
# QR-2 (QR 50%)
python scripts/lightgcn/exp_find_hparams.py -c configs/lightgcn/qr-2.yaml -l logs/qr-2 -p checkpoints/lightgcn/qr-2.pth

# QR-5 (QR 80%)
python scripts/lightgcn/exp_find_hparams.py -c configs/lightgcn/qr-5.yaml -l logs/qr-5 -p checkpoints/lightgcn/qr-5.pth

# OptEmbed----------------------
# supernet
python scripts/lightgcn/exp_find_hparams.py -c configs/lightgcn/opt-embed.yaml -l logs/opt-embed -p checkpoints/lightgcn/opt-embed.pth
# evol
cp checkpoints/lightgcn/opt-embed.pth checkpoints/checkpoint.pth
cp configs/best-trial.yaml configs/best-opt-embed-super.yaml
python scripts/lightgcn/run_opt_evol_lightgcn.py configs/lightgcn/opt-embed.yaml

# retrain
python scripts/generate_retrain_config.py configs/best-opt-embed-super.yaml configs/opt-embed-retrain.yaml
python scripts/lightgcn/train_lightgcn.py configs/lightgcn/opt-embed-retrain.yaml


# PEP------


# L2



# TT-Rec
python scripts/exp_find_hparams.py -c configs/lightgcn/tt-emb-96-96.yaml -l logs/lightgcn/tt-emb-96-96 -p checkpoints/lightgcn/tt-emb-96-96.pth
# DHE
# Quantization
