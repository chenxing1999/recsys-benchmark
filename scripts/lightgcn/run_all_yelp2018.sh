mkdir -p checkpoints/lightgcn/

# SGL-WA base config
python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/base_config.yaml \
    -l logs/sgl-wa-base/ \
    -p checkpoints/lightgcn/sgl-wa.pth
cp configs/best-trial.yaml configs/yelp2018/sgl-wa-best-config.yaml

# QR----------------------------
# QR-2 (QR 50%)
python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/qr-2.yaml -l logs/qr-2 -p checkpoints/lightgcn/qr-2.pth

# QR-5 (QR 80%)
python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/qr-5.yaml -l logs/qr-5 -p checkpoints/lightgcn/qr-5.pth
#
# OptEmbed----------------------
# supernet
#--- create checkpoint at checkpoints/lightgcn/opt-embed.pth and config at configs/best-trial.yaml
python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/opt-embed.yaml \
    -l logs/opt-embed \
    -p checkpoints/lightgcn/opt-embed.pth

# evol
cp checkpoints/lightgcn/opt-embed.pth checkpoints/opt-embed-super.pth
cp configs/best-trial.yaml configs/yelp2018/best-opt-embed-super.yaml
python scripts/generate_config.py configs/yelp2018/best-opt-embed-super.yaml configs/yelp2018/best-opt-embed-evol.yaml --run_test_mode False
python scripts/lightgcn/run_opt_evol_lightgcn.py configs/yelp2018/best-opt-embed-evol.yaml

# retrain
python scripts/generate_config.py \
    configs/best-opt-embed-evol.yaml \
    configs/yelp2018/opt-embed-retrain.yaml \
    --checkpoint_path checkpoints/lightgcn/best-opt-retrain.pth \
    --run_test_mode False \
    --add-retrain

python scripts/lightgcn/train_lightgcn.py configs/yelp2018/opt-embed-retrain.yaml


# OptEmbed-100epochs----------------------
# supernet
# python scripts/lightgcn/exp_find_hparams.py -c configs/yelp2018/opt-embed-100.yaml -l logs/opt-embed-100 -p checkpoints/lightgcn/opt-embed-100.pth
# evol
# cp checkpoints/lightgcn/opt-embed.pth checkpoints/checkpoint.pth
# cp checkpoints/lightgcn/opt-embed.pth checkpoints/opt-embed-super.pth
# cp configs/best-trial.yaml configs/yelp2018/best-opt-embed-super.yaml
# python scripts/lightgcn/run_opt_evol_lightgcn.py configs/yelp2018/best-opt-embed-super.yaml
#
# retrain
# python scripts/generate_retrain_config.py configs/best-opt-embed-super.yaml configs/yelp2018/opt-embed-retrain.yaml
# python scripts/lightgcn/train_lightgcn.py configs/yelp2018/opt-embed-retrain.yaml

# PEP------


# L2
echo "L2 --- 0.8"
python scripts/lightgcn/run_l2_benchmark.py configs/yelp2018/sgl-wa-best-config.yaml -c checkpoints/lightgcn/sgl-wa.pth -p 0.8
echo "L2 --- 0.5"
python scripts/lightgcn/run_l2_benchmark.py configs/yelp2018/sgl-wa-best-config.yaml -c checkpoints/lightgcn/sgl-wa.pth -p 0.5


# TT-Rec
python scripts/exp_find_hparams.py -c configs/yelp2018/tt-emb-96-96.yaml -l logs/lightgcn/tt-emb-96-96 -p checkpoints/lightgcn/tt-emb-96-96.pth

# DHE (80%)
python scripts/exp_find_hparams.py -c configs/yelp2018/dhe-256-256-128-128.yaml -l logs/lightgcn/dhe-80 -p checkpoints/lightgcn/dhe-80.pth

# Quantization
