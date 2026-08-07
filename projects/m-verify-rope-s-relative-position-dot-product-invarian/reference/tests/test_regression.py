import numpy as np
import sys
sys.path.insert(0, ".")

from rope.core import compute_rope_frequencies, apply_rope, apply_position_interpolation


def test_position_interpolation_rescaling():
    dim = 64
    max_len = 8192
    scale_factor = 4.0
    freqs = compute_rope_frequencies(dim, max_len)

    q = np.random.randn(dim)
    k = np.random.randn(dim)

    pos_orig = 1000
    pos_scaled = apply_position_interpolation(pos_orig, scale_factor)

    assert not np.isclose(pos_orig, pos_scaled), "Position interpolation should scale position indices."

    q_rot_unscaled = apply_rope(q, pos_orig, freqs)
    q_rot_scaled = apply_rope(q, int(pos_scaled), freqs)

    diff = np.max(np.abs(q_rot_unscaled - q_rot_scaled))
    assert diff > 1e-3, "Scaled position must produce different embeddings than original position."
