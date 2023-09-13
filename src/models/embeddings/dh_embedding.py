import json
import os
from typing import Final, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .base import IEmbedding

LARGE_INT: Final[int] = int(1e9)
NEGATIVE_LARGE_INT: Final[int] = -LARGE_INT


class DHEmbedding(IEmbedding):
    """DHE Hashing method proposed in
    https://arxiv.org/pdf/2010.10784.pdf"""

    COUNTER = 0

    def __init__(
        self,
        field_dims: Union[int, List[int]],
        out_size: int,
        mode: Optional[str] = None,
        inp_size: int = 1024,
        hidden_sizes: Optional[List[int]] = None,
        use_bn: bool = True,
        cached: bool = True,
        prime_file: Optional[str] = None,
    ):
        """
        Args:
            field_dims: Categorical feature sizes (it is here to keep API format)
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

        if isinstance(field_dims, int):
            field_dims = [field_dims]

        num_item = sum(field_dims)
        self.m = int(1e6)

        # Using prefix to make sure that embedding of
        # user and item are different
        self._prefix = DHEmbedding.COUNTER
        DHEmbedding.COUNTER += num_item

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

        self._mode = mode

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
        k = self._inp_size

        # Setting torch manual seed as item instead of 0
        # to allow retrieving item hash in a function
        torch.manual_seed(item + self._prefix)

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
        """Initialize hash value for all items and store it in cache"""
        cache = []
        for item in range(self._num_item):
            cache.append(self._get_hash(item))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        cache = torch.stack(cache)
        cache = cache.to(device)
        self._cache = cache

    def forward(self, inp: Union[torch.IntTensor, torch.LongTensor]):
        device = self._seq[0].weight.device
        mode = self._mode
        if self._use_cache:
            cache = self._cache
            cache = cache.to(device)

            if mode is None:
                embs = F.embedding(inp, cache)
            else:
                embs = F.embedding_bag(inp, cache, mode=mode)
        else:
            vecs = [self._get_hash(item) for item in inp]
            embs = torch.tensor(vecs, device=device)

            # Not implement inplace operation like torch.Bag
            if mode is None:
                pass
            elif mode == "sum":
                embs = embs.sum(1)
            elif mode == "max":
                embs = embs.max(1)
            elif mode == "mean":
                embs = embs.mean(1)
            else:
                raise NotImplementedError()

        is_flatten = False
        if len(embs.shape) == 3:
            is_flatten = True
            batch, num_field, dimension = embs.shape
            embs = embs.reshape(batch * num_field, dimension)

        outs = self._seq(embs)
        if is_flatten:
            outs = outs.reshape(batch, num_field, -1)

        return outs

    def set_extra_state(self, state):
        self._prefix = state["_prefix"]

    def get_extra_state(self):
        return {"_prefix": self._prefix}
