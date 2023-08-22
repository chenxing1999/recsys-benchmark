import copy
import json
import math
import os
from typing import Dict, List, Literal, Optional, Union

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F


class IEmbedding(nn.Module):
    """
    Note: Not all weight is initialized from the same distribution
    as I found out that the Xavier uniform despite converge slower
    but usually get higher accuracy.
    """

    def get_weight(self):
        ...


class VanillaEmbedding(nn.Embedding, IEmbedding):
    """Wrapper for default PyTorch Embedding"""

    def __init__(self, *args, initializer="xavier", **kwargs):
        super().__init__(*args, **kwargs)

        if initializer == "xavier":
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.normal_(self.weight, std=0.1)

    def get_weight(self):
        return self.weight


class QRHashingEmbedding(IEmbedding):
    """QUOTIENT-REMAINDER based hasing Embedding
    Original paper: https://arxiv.org/pdf/1909.02107.pdf
    """

    def __init__(
        self,
        num_item: int,
        hidden_size: int,
        divider: Optional[int] = None,
        operation: Literal["cat", "add", "mult"] = "mult",
        initializer="uniform",
    ):
        """
        Args:
            num_item: Categorical feature size
            hidden_size: Output embedding hidden size.
                For simplicity, we dont allow different size for concatenate feature

            divider:
            operation: Support:
                cat (concatenate): Concatenating feature
                add: Adding feature to each other
                mult: Element wise multiplication
        """

        super().__init__()
        assert operation in ["cat", "add", "mult"]
        if operation == "cat":
            assert hidden_size % 2 == 0

        if divider is None:
            divider = int(math.sqrt(num_item))

        emb_size = hidden_size
        if operation == "cat":
            emb_size = hidden_size // 2

        self._operation = operation

        size = (num_item - 1) // divider + 1
        self.emb1 = nn.Embedding(divider, emb_size)
        self.emb2 = nn.Embedding(size, emb_size)
        self._hidden_size = hidden_size
        self._divider = divider
        self._num_item = num_item

        if initializer == "normal":
            self._init_normal_weight()
        elif initializer == "uniform":
            self._init_uniform_weight()

    def _init_uniform_weight(self):
        # Haven't found method to generate emb1 and emb2
        # so that ops(emb1, emb2) will have uniform(-a, a).
        # Note: Orignal QR init method use uniform(sqrt(1 / num_categories), 1)
        # https://github.com/facebookresearch/dlrm/blob/dee060be6c2f4a89644acbff6b3a36f7c0d5ce39/tricks/qr_embedding_bag.py#L182-L183
        # But I use uniform(-alpha, alpha) with alpha = sqrt(1 / num_categories)

        alpha = math.sqrt(1 / self._num_item)
        nn.init.uniform_(self.emb1.weight, alpha, 1)
        nn.init.uniform_(self.emb2.weight, alpha, 1)

    def _init_normal_weight(self):
        std = 0.1
        if self._operation == "add":
            std = std / 2
        elif self._operation == "mult":
            std = math.sqrt(std)
        nn.init.normal_(self.emb1.weight, std=std)
        nn.init.normal_(self.emb2.weight, std=std)

    def forward(self, tensor: torch.IntTensor):
        inp1 = tensor % self._divider
        inp2 = tensor // self._divider

        emb1 = self.emb1(inp1)
        emb2 = self.emb2(inp2)

        if self._operation == "cat":
            return torch.cat([emb1, emb2], dim=1)
        elif self._operation == "add":
            return emb1 + emb2
        elif self._operation == "mult":
            return emb1 * emb2
        else:
            raise NotImplementedError()

    def get_weight(self):
        arr = torch.arange(self._num_item, device=self.emb1.weight.device)
        return self(arr)


class DHEEmbedding(IEmbedding):
    """DHE Hashing method proposed in
    https://arxiv.org/pdf/2010.10784.pdf"""

    def __init__(
        self,
        num_item: int,
        out_size: int,
        inp_size: int = 1024,
        hidden_sizes: Optional[List[int]] = None,
        use_bn: bool = True,
        cached: bool = True,
        prime_file: Optional[str] = None,
    ):
        """
        Args:
            num_item: Number of embedding item (it is here to keep API format)
            out_size: Last layer output size (it is here to keep API format)

            inp_size: Input size / `k` in the paper
            hidden_sizes: List[int] for hidden size of each table
            use_bn: Use batchnorm or not
            cached: Will store all hash vector in memory or run online

            prime_file: Contains list of possible value for p
                (prime number precomputed that are larger than 1e6)
        """
        super().__init__()

        if prime_file is None:
            prime_file = os.path.join(
                os.path.dirname(__file__), "../../assets/large_prime_74518.json"
            )

        self.m = int(1e6)
        with open(prime_file) as fin:
            primes = json.load(fin)

        self._primes = torch.tensor(primes)

        layers = []
        self._inp_size = inp_size
        self._num_item = num_item

        if hidden_sizes is None:
            hidden_sizes = []

        hidden_sizes.append(out_size)
        for size in hidden_sizes:
            layers.append(nn.Linear(inp_size, size))
            layers.append(nn.Mish())
            if use_bn:
                layers.append(nn.BatchNorm1d(size))

            inp_size = size

        self._seq = nn.Sequential(*layers)
        self._cache: Union[List[None], torch.Tensor] = [None] * self._num_item
        self._use_cache = cached
        self._use_bn = use_bn

        if cached:
            self._init_all_hash()

    def get_weight(self):
        arr = torch.arange(
            self._num_item,
            device=self._seq[0].weight.device,
        )
        return self(arr)

    def _get_hash(self, item: int) -> torch.Tensor:
        """Get Hash value of a single item"""

        if self._cache[item] is not None:
            return self._cache[item]

        m = self.m

        primes = self._primes
        LARGE_INT = int(1e9)
        NEGATIVE_LARGE_INT = -LARGE_INT
        k = self._inp_size

        # Setting torch manual seed as item instead of 0
        # to allow retrieving item hash in a function
        torch.manual_seed(item)

        a = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (k,))
        b = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (k,))

        mask = b == 0
        while mask.sum() > 0:
            b[mask] = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (mask.sum(),))
            mask = b == 0

        p_idx = torch.randint(0, len(primes), (k,))
        primes_choices = primes[p_idx]

        hidden_values = (a * (item + 1) + b) % primes_choices % m

        # Only implement uniform as paper stated there should
        # be not much different between Gaussian and Uniform
        encod = hidden_values / (m - 1)
        encod = encod * 2 - 1
        return encod

    def _init_all_hash(self):
        cache = []
        for item in range(self._num_item):
            cache.append(self._get_hash(item))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        cache = torch.stack(cache)
        cache = cache.to(device)
        self._cache = cache

    def forward(self, inp: torch.IntTensor):
        device = self._seq[0].weight.device
        if self._use_cache:
            cache = self._cache
            cache = cache.to(device)
            embs = torch.index_select(cache, 0, inp)
        else:
            vecs = [self._get_hash(item) for item in inp]
            embs = torch.tensor(vecs, device=device)

        outs = self._seq(embs)

        # This seems not working ...
        # if self._use_bn:
        #     # Normalize factor to make sure std to 0.1
        #     # If not use BN, I cannot control the exact std anyways
        #     # So I will ignore that case
        #     outs = outs * 0.1
        return outs


def get_embedding(
    embedding_config: Dict,
    num_item: int,
    hidden_size: int,
    field_name: str = "",
) -> IEmbedding:
    name = embedding_config["name"]
    embedding_config = copy.deepcopy(embedding_config)
    embedding_config.pop("name")

    name_to_cls = {
        "vanilla": VanillaEmbedding,
        "qr": QRHashingEmbedding,
        "dhe": DHEEmbedding,
        "pep": PepEmbeeding,
        "pep_retrain": RetrainPepEmbedding,
    }
    if name == "vanilla":
        emb = VanillaEmbedding(
            num_item,
            hidden_size,
            **embedding_config,
        )
    elif name == "optembed":
        from src.models.opt_embed import OptEmbed

        emb = OptEmbed(num_item, hidden_size)
    elif name not in name_to_cls:
        raise NotImplementedError()
    else:
        if name.startswith("pep"):
            embedding_config["field_name"] = field_name
        cls = name_to_cls[name]
        emb = cls(num_item, hidden_size, **embedding_config)

    embedding_config["name"] = name
    return emb


class PepEmbeeding(IEmbedding):
    """PEP Embedding proposed in
        https://arxiv.org/pdf/2101.07577.pdf

    Original implementation:
        https://github.com/ssui-liu/learnable-embed-sizes-for-RecSys
    """

    def __init__(
        self,
        num_item: int,
        hidden_size: int,
        ori_weight_dir: str,
        checkpoint_weight_dir: str,
        field_name: str = "",
        init_threshold: float = -150,
        threshold_type: str = "feature_dim",
        sparsity: Optional[List[float]] = None,
    ):
        """
        Args:
            num_item
            hidden_size
            ori_weight_path: Path to original weight for later retrain
                with Lottery Ticket
            init_threshold: Initialize value for s
            threshold_type: Support `feature_dim`, `feature`, `dimension` and `global
            sparsity
        """
        super().__init__()

        if sparsity is None:
            sparsity = [0.8, 0.9, 0.99]
        self.sparsity = list(sorted(sparsity))
        self._cur_min_spar_idx = 0

        # Initialize and Save the original weight to get winning ticket later
        self.emb = nn.Embedding(num_item, hidden_size)
        nn.init.xavier_uniform_(self.emb.weight)

        os.makedirs(ori_weight_dir, exist_ok=True)
        ori_weight_path = os.path.join(ori_weight_dir, field_name + ".pth")
        torch.save({"state_dict": self.emb.state_dict()}, ori_weight_path)

        self.threshold_type = threshold_type
        self.s = self.init_threshold(init_threshold, num_item, hidden_size)

        self.field_name = field_name
        if field_name:
            checkpoint_weight_dir = os.path.join(checkpoint_weight_dir, field_name)
        os.makedirs(checkpoint_weight_dir, exist_ok=True)
        self.checkpoint_weight_dir = checkpoint_weight_dir

    def get_weight(self):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        return sparse_weight

    def forward(self, x):
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        xv = F.embedding(x, sparse_weight)

        return xv

    def soft_threshold(self, v, s):
        return torch.sign(v) * torch.relu(torch.abs(v) - torch.sigmoid(s))

    def init_threshold(self, init, num_item, hidden_size) -> nn.Parameter:
        """

        threshold_type
            global: single threshold for all item
            dimension: threshold for all dimension


        """
        if self.threshold_type == "global":
            s = nn.Parameter(init * torch.ones(1))
        elif self.threshold_type == "dimension":
            s = nn.Parameter(init * torch.ones([hidden_size]))
        elif self.threshold_type == "feature":
            s = nn.Parameter(init * torch.ones([num_item, 1]))
        elif self.threshold_type == "field":
            raise NotImplementedError()
        elif self.threshold_type == "feature_dim":
            s = nn.Parameter(init * torch.ones([num_item, hidden_size]))
        elif self.threshold_type == "field_dim":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid threshold_type: {}".format(self.threshold_type))
        return s

    def get_sparsity(self) -> float:
        total_params = self.emb.weight.numel()
        sparse_weight = self.soft_threshold(self.emb.weight, self.s)
        n_params = (sparse_weight != 0).sum()
        return (1 - n_params / total_params).item()

    def train_callback(self):
        """Callback to save weight to `checkpoint_weight_dir`"""
        with torch.no_grad():
            cur_sparsity = self.get_sparsity()

        while (
            self._cur_min_spar_idx < len(self.sparsity)
            and self.sparsity[self._cur_min_spar_idx] < cur_sparsity
        ):
            sparsity = self.sparsity[self._cur_min_spar_idx]
            logger.info(f"cur_sparsity is larger than {sparsity}")

            # Save model
            path = os.path.join(self.checkpoint_weight_dir, f"{sparsity}.pth")
            torch.save(self.state_dict(), path)
            self._cur_min_spar_idx += 1


class RetrainPepEmbedding(IEmbedding):
    """Wrapper for Retrain inference logic of Pep Embedding"""

    def __init__(
        self,
        num_item: int,
        hidden_size,
        checkpoint_weight_dir,
        sparsity: Union[float, str],
        ori_weight_dir: Optional[str] = None,
        field_name: str = "",
    ):
        """
        Args:
            num_item
            hidden_size
            checkpoint_weight_dir: Path pep_checkpoint folder
                The weight to get weight mask from should be
                    f"{checkpoint_weight_dir}/{field_name}/{sparsity}.pth

            sparsity: Target sparsity
            ori_weight_dir: Path to original weight
                if not provided, you could try to manually load the original weight
            field_name: Name to field (used to get checkpoint mask path)
        """
        super().__init__()
        self.emb = nn.Embedding(num_item, hidden_size)

        if ori_weight_dir:
            ori_weight_path = os.path.join(ori_weight_dir, field_name + ".pth")
            original_state_dict = torch.load(ori_weight_path, map_location="cpu")[
                "state_dict"
            ]
            self.emb.load_state_dict(original_state_dict)

        finish_weight_path = os.path.join(
            checkpoint_weight_dir,
            field_name,
            f"{sparsity}.pth",
        )
        finish_weight = torch.load(finish_weight_path, map_location="cpu")
        weight = finish_weight["emb.weight"]
        s = finish_weight["s"]
        self.mask = (torch.abs(weight) - torch.sigmoid(s)) > 0
        self.mask = self.mask.cuda()

        nnz = self.mask.sum()
        self.sparsity = 1 - (nnz / torch.prod(torch.tensor(self.mask.size()))).item()

    def get_weight(self):
        sparse_emb = self.emb.weight * self.mask
        return sparse_emb

    def forward(self, x):
        sparse_emb = self.emb.weight * self.mask
        return F.embedding(x, sparse_emb)

    def get_sparsity(self):
        return self.sparsity
