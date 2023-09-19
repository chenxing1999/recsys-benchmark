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
python scripts/lightgcn/exp_find_hparams.py
