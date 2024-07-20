import torch
from torch.nn import functional as F

from src.models.embeddings import IEmbedding


class PTQEmb_Fp16(IEmbedding):
    """Post-training Quantization Embedding Float16"""

    def __init__(self, field_dims, num_factor, mode, ori_checkpoint_path):
        """
        Args:
            field_dims, num_factor: Could be anything, only there for API
                format
        """
        super().__init__()

        checkpoint = torch.load(ori_checkpoint_path)
        state = checkpoint["state_dict"]
        emb = state["embedding._emb_module.weight"]
        # Convert emb to
        self.register_buffer("weight", emb.to(torch.float16))

    def forward(self, x):
        return F.embedding(x, self.weight).to(torch.float32)

    def get_weight(self) -> torch.Tensor:
        return self.weight

    def get_num_params(self) -> int:
        return self.weight.shape[0] * self.weight.shape[1]


class PTQEmb_Int(IEmbedding):
    """Post-training Quantization Embedding Int8"""

    def __init__(
        self,
        field_dims,
        num_factor,
        mode,
        ori_checkpoint_path,
        n_bits=8,
    ):
        """
        Args:
            field_dims, num_factor: Could be anything, only there for API
                format

            ori_checkpoint_path: Path to original checkpoint to
                quantize
            n_bits: Num bits, 8 or 16
        """
        super().__init__()

        checkpoint = torch.load(ori_checkpoint_path)
        state = checkpoint["state_dict"]
        emb: torch.Tensor = state["embedding._emb_module.weight"]

        # calculate scale
        assert n_bits in [8, 16]
        self.n_bits = n_bits

        q_min = (-1) * (1 << (self.n_bits - 1))
        q_max = (1 << (self.n_bits - 1)) - 1
        r_min = emb.min().cpu()
        scale = (emb.max().item() - r_min) / (q_max - q_min)

        dtype = torch.int8
        if self.n_bits == 16:
            dtype = torch.int16

        bias = (q_min - r_min / scale).to(dtype)

        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

        weight = emb / scale + bias
        torch.round_(weight)
        self.register_buffer("weight", weight.to(dtype))

    def forward(self, x):
        res = F.embedding(x, self.weight)

        return (res - self.bias) * self.scale

    def get_weight(self) -> torch.Tensor:
        return (self.weight - self.bias) * self.scale

    def get_num_params(self) -> int:
        return self.weight.shape[0] * self.weight.shape[1]


if __name__ == "__main__":
    from src.models.deepfm import DeepFM

    checkpoint = "/home/xing/workspace/phd/run-code/checkpoints/deepfm/original.pth"

    model = DeepFM.load(checkpoint)
    emb1 = model.embedding
    print(emb1.get_weight().mean())
    emb2 = PTQEmb_Int(None, None, None, checkpoint)
