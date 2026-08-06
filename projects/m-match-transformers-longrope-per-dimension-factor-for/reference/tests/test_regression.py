import numpy as np
from longrope.scaling import compute_longrope_factors


def test_longrope_non_uniform_scaling():
    head_dim = 64
    orig_len = 4096
    target_len = 16384

    factors = compute_longrope_factors(head_dim, orig_len, target_len)

    assert len(factors) == head_dim // 2
    assert not np.allclose(factors, factors[0]), "Scaling factors must not be uniform across all dimensions"
    assert np.all(factors >= 1.0), "Scaling factors must be >= 1.0"
    assert np.all(factors <= (target_len / orig_len)), "Scaling factors must not exceed maximum scale ratio"
