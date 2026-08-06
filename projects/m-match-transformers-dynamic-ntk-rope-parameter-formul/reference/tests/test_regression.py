import sys
import numpy as np

sys.path.insert(0, ".")
from ropescaling.scaling import compute_dynamic_ntk_base, compute_yarn_parameters, compute_llama3_scaling


def test_dynamic_ntk_scaling_expands():
    base = 10000.0
    res = compute_dynamic_ntk_base(base, 8192, 4096)
    assert res > base


def test_yarn_output_shape_and_mscale():
    freqs, mscale = compute_yarn_parameters(10000.0, 8192, 4096, 4096, 32.0, 1.0, 1.0)
    assert isinstance(freqs, np.ndarray)
    assert freqs.shape == (64,)
    assert mscale >= 1.0


def test_llama3_scaling_modifies_frequencies():
    freqs = compute_llama3_scaling(10000.0, 8192, 4096, 4096, 2.0, 1.0, 4.0)
    assert isinstance(freqs, np.ndarray)
    assert freqs.shape == (64,)
