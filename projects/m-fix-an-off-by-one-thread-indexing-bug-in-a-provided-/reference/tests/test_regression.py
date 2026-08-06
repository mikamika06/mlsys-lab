import sys
sys.path.insert(0, ".")
from metal_kernels.kernel import run_indexing_kernel, run_sum_reduction_kernel, run_dequant_kernel
import numpy as np

def test_indexing_bounds():
    res = run_indexing_kernel(128)
    assert res is not None
    assert len(res) == 128
    assert res[0] == 1.0
    assert res[-1] == 128.0

def test_sum_reduction_consistency():
    arr = np.ones(100, dtype=np.float32)
    s_safe = run_sum_reduction_kernel(arr, math_mode="safe")
    assert np.isclose(s_safe, 100.0)

def test_dequant_output():
    packed = np.array([0x12, 0x34], dtype=np.uint8)
    scales = np.array([1.0], dtype=np.float32)
    biases = np.array([0.0], dtype=np.float32)
    res = run_dequant_kernel(packed, scales, biases)
    assert res is not None
    assert res.shape[-1] == 4
    assert res[0, 0] == 2.0
    assert res[0, 1] == 1.0
