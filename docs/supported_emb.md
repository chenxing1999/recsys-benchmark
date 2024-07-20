# Compositional-based Embedding

## QR

Paper: [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems](https://arxiv.org/pdf/1909.02107)
[code link](../src/models/embeddings/qr_embedding.py)

Only supported two tables for simplicity.

## TTRec

Paper:
[code link](../src/models/embeddings/tensortrain_embeddings.py)

- Provided both CUDA-based implementation by the authors (`TTEmbedding`) and PyTorch reimplementation (`TTRecTorch`).
- CUDA version is faster, support cache. However, it requires to compile a custom CUDA kernel.
- PyTorch implementation is much more easy to install and play with. However, it is slower and doesn't support cache

## DHE

# Pruning

## PEP

## OptEmbed

# Hybrid

## CERP

# Quantization

## QAT-Aware with Stochastic Rounding

## Post-Training Quantization
