import json
import os
from typing import Final, List, Optional, Union

import torch
from loguru import logger
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
        use_bn: Union[bool, int] = 2,
        cached: bool = True,
        prime_file: Optional[str] = None,
        cache_path: str = "",
        compute_v2=False,
        use_universal_hash=True,
    ):
        """
        Args:
            field_dims: Categorical feature sizes (it is here to keep API format)
            out_size: Last layer output size (it is here to keep API format)

            inp_size: Input size / `k` in the paper
            hidden_sizes: List[int] for hidden size of each table
            use_bn: Use batchnorm or not
                use_bn == (1, True) to use Linear -> Mish -> BatchNorm
                (parameter name is not changed to make sure old weight still can load)
            cached: Will store all hash vector in memory or run online

            prime_file: Contains list of possible value for p
                (prime number precomputed that are larger than 1e6)

            cache_path: Path to hashed value initialization
                Note: not supported for LightGCN / multiple embedding tables currently

            compute_v2:
                Run embedding -> Unique -> DHE -> Reverse Unique -> result
                (original flow: Embedding -> DHE -> Result)

            use_universal_hash:
                Using the correct implementation of universal hash
        """
        super().__init__()

        if prime_file is None:
            prime_file = os.path.join(
                os.path.dirname(__file__), "../../assets/large_prime_74518.json"
            )

        if isinstance(field_dims, int):
            field_dims = [field_dims]

        if isinstance(use_bn, bool):
            use_bn = int(use_bn)

        num_item = sum(field_dims)
        self.m = int(1e6)

        # Using prefix to make sure that embedding of
        # user and item are different
        self._prefix = DHEmbedding.COUNTER
        DHEmbedding.COUNTER += num_item

        with open(prime_file) as fin:
            primes = json.load(fin)

        self._primes = torch.tensor(primes)
        self._inp_size = inp_size
        self._num_item = num_item

        self._use_universal_hash = use_universal_hash
        if use_universal_hash:
            rng = torch.Generator()
            rng.manual_seed(0)
            self.register_buffer("_slopes", self._random_nonzero_int(inp_size, rng))
            self.register_buffer("_bias", self._random_nonzero_int(inp_size, rng))
            p_idx = torch.randint(0, len(primes), (inp_size,), generator=rng)

            self.register_buffer("_primes_choices", self._primes[p_idx])
        layers: List[nn.Module] = []

        if hidden_sizes is None:
            hidden_sizes = []

        hidden_sizes.append(out_size)
        for size in hidden_sizes:
            layers.append(nn.Linear(inp_size, size))
            if use_bn == 1:
                layers.append(nn.Mish())
                layers.append(nn.BatchNorm1d(size))
            elif use_bn == 2:
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.Mish())
            else:
                layers.append(nn.Mish())

            inp_size = size

        # self._seq = torch.compile(nn.Sequential(*layers))
        self._seq = nn.Sequential(*layers)
        self._cache: Union[List[None], torch.Tensor] = [None] * self._num_item
        self._use_cache = cached
        self._use_bn = use_bn
        self._out_size = out_size

        self.compute_v2 = compute_v2

        self._mode = mode
        logger.debug(f"Num params: {self.get_num_params()}")

        if cached:
            if os.path.exists(cache_path):
                self._cache = torch.load(cache_path)
            else:
                self._init_all_hash()
                if cache_path:
                    torch.save(self._cache, cache_path)

        self._emb = None

    def get_weight(self):
        device = self._seq[0].weight.device
        # device = "cuda"
        arr = torch.arange(
            self._num_item,
            device=device,
        )
        return self(arr)

    def _get_hash(self, item: int) -> torch.Tensor:
        """Get Hash value of a single item"""

        if self._use_universal_hash:
            return self._get_universal_hash(item)

        if self._cache[item] is not None:
            return self._cache[item]

        m = self.m

        primes = self._primes
        k = self._inp_size

        # Setting torch manual seed as item instead of 0
        # to allow retrieving item hash in a function
        torch.manual_seed(item + self._prefix)
        rng = None
        # torch.manual_seed(0)
        # rng = torch.Generator()
        # rng.manual_seed(0)
        # rng.manual_seed(self._prefix)
        # item = item + self._prefix

        a = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (k,), generator=rng)
        b = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (k,), generator=rng)

        mask = b == 0
        while mask.sum() > 0:
            num_values = mask.sum().item()
            b[mask] = torch.randint(
                NEGATIVE_LARGE_INT, LARGE_INT, (num_values,), generator=rng
            )
            mask = b == 0

        p_idx = torch.randint(0, len(primes), (k,), generator=rng)
        primes_choices = primes[p_idx]

        hidden_values = (a * (item + 1) + b) % primes_choices % m

        # Only implement uniform as paper stated there should
        # be not much different between Gaussian and Uniform
        encod = hidden_values / (m - 1)
        encod = encod * 2 - 1
        return encod

    def _get_universal_hash(self, item: int):
        """Universal hash function"""

        if self._cache[item] is not None:
            return self._cache[item]

        m = self.m

        primes_choices = self._primes_choices
        item = item + self._prefix

        hidden_values = (self._slopes * (item + 1) + self._bias) % primes_choices % m

        # Only implement uniform as paper stated there should
        # be not much different between Gaussian and Uniform
        encod = hidden_values / (m - 1)
        encod = encod * 2 - 1
        return encod

    def _get_universal_hash_batch(self, item: torch.tensor):
        device = item.device
        slopes, bias, primes = self._slopes, self._bias, self._primes_choices

        slopes, bias, primes = slopes.to(device), bias.to(device), primes.to(device)

        slopes = slopes.unsqueeze(0)
        bias = bias.unsqueeze(0)
        primes = primes.unsqueeze(0)
        item = item.unsqueeze(1)

        BATCH_SIZE = int(1e5)

        res = []
        for start in range(0, len(item), BATCH_SIZE):
            tmp_item = item[start : start + BATCH_SIZE]
            result = slopes * (tmp_item + self._prefix + 1) + bias
            result = result % primes % self.m
            result = result / (self.m - 1)
            res.append(result)
        result = torch.cat(res)
        result = result * 2 - 1

        return result

    def _random_nonzero_int(self, num_element, rng=None):
        b = torch.randint(NEGATIVE_LARGE_INT, LARGE_INT, (num_element,), generator=rng)

        mask = b == 0
        while mask.sum() > 0:
            num_values = mask.sum().item()
            b[mask] = torch.randint(
                NEGATIVE_LARGE_INT, LARGE_INT, (num_values,), generator=rng
            )
            mask = b == 0
        return b

    def _init_all_hash(self):
        """Initialize hash value for all items and store it in cache"""

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        # if self._use_universal_hash:
        #     item = torch.arange(self._num_item, device=device, dtype=int)
        #     self._cache = self._get_universal_hash_batch(item)
        #     return

        cache = []
        for item in range(self._num_item):
            cache.append(self._get_hash(item))

        cache = torch.stack(cache)
        cache = cache.to(device)
        self._cache = cache

    def forward(self, inp: Union[torch.IntTensor, torch.LongTensor]):
        device = self._seq[0].weight.device
        # device = "cuda"
        mode = self._mode
        if not self.training and self._emb is not None:
            return F.embedding(
                inp,
                self._emb,
            )

        if self.compute_v2 and self._use_cache:
            uniques, inverse_idx = inp.unique(return_inverse=True)
            cache = self._cache
            if cache.device != device:
                logger.warning(
                    "Cache device ({cache.device})"
                    "is different from MLP device ({device})"
                )
            cache = cache.to(device)

            x = F.embedding(uniques, cache)
            x = self._forward_mlp(x)
            result = x[inverse_idx]
            return result

        if self._use_cache and isinstance(self._cache, torch.Tensor):
            cache = self._cache
            if cache.device != device:
                logger.warning(
                    f"Cache device ({cache.device})"
                    f"is different from MLP device ({device})"
                )
            cache = cache.to(device)

            if mode is None:
                embs = F.embedding(inp, cache)
            else:
                embs = F.embedding_bag(inp, cache, mode=mode)
        else:
            # init emb
            uniques, inverse_idx = inp.unique(return_inverse=True)

            # embs = torch.tensor(vecs, device=device)
            if not self.training:
                if self._use_universal_hash:
                    feats = self._get_universal_hash_batch(uniques)
                else:
                    uniques = uniques.tolist()
                    feats: torch.Tensor = torch.stack(
                        [self._get_hash(v) for v in uniques]
                    )
                feats = feats.to(device)
                feats = self._seq(feats)
                # tmp_emb_size = max(inverse_idx) + 1

                # tmp_emb = torch.zeros((tmp_emb_size, self._out_size), device=device)
                # tmp_emb[inverse_idx] = feats
                return F.embedding(inverse_idx, feats)
            else:
                raise NotImplementedError()

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

        return self._forward_mlp(embs)

    def _forward_mlp(self, embs):
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
