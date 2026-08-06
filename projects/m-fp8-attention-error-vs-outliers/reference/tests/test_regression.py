import sys
import numpy as np

sys.path.insert(0, ".")
from fp8att.hadamard import apply_hadamard_transform
from fp8att.analysis import compute_attention_error


def test_hadamard_preserves_norm():
    x = np.random.default_rng(0).normal(0.0, 1.0, (1, 1, 16, 64)).astype(np.float32)
    x[:, :, :, 5] *= 50.0
    out = apply_hadamard_transform(x)
    norm_orig = np.linalg.norm(x)
    norm_out = np.linalg.norm(out)
    assert abs(norm_orig - norm_out) / (norm_orig + 1e-8) < 1e-5


def test_hadamard_reduces_max_magnitude():
    x = np.random.default_rng(1).normal(0.0, 1.0, (1, 1, 16, 64)).astype(np.float32)
    x[:, :, :, 5] *= 100.0
    max_orig = np.max(np.abs(x))
    out = apply_hadamard_transform(x)
    max_out = np.max(np.abs(out))
    assert max_out < max_orig
