import numpy as np
import pytest

from src.models.embeddings.optembed_utils import _get_expected_hidden_size


@pytest.mark.parametrize("alpha", [0.983, 1.083])
@pytest.mark.parametrize("hidden_size", [8, 16, 64])
def test_calc_e(alpha, hidden_size):
    # Calculate through step by step
    f = np.power(alpha, np.arange(1, hidden_size + 1) * (-1) + hidden_size)
    p = f / f.sum()
    e = (np.arange(1, hidden_size + 1) * p).sum()

    # approx
    approx_e = _get_expected_hidden_size(alpha, hidden_size)

    assert abs(e - approx_e) < 1e-6
