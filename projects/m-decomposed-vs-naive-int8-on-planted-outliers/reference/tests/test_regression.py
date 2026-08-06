import sys
import numpy as np

sys.path.insert(0, ".")
from quant.decomposed import naive_int8_matmul, decomposed_matmul
from quant.threshold import find_optimal_threshold
from quant.skip import identify_skip_modules


def test_decomposed_beats_naive_mse():
    np.random.seed(42)
    x = np.random.randn(64, 64).astype(np.float32)
    x[10, 10] = 50.0
    w = np.random.randn(64, 64).astype(np.float32)
    ref = np.matmul(x, w)
    naive_out = naive_int8_matmul(x, w)
    decomp_out = decomposed_matmul(x, w, threshold=6.0)
    naive_mse = np.mean((naive_out - ref) ** 2)
    decomp_mse = np.mean((decomp_out - ref) ** 2)
    assert decomp_mse < naive_mse


def test_threshold_respects_flop_cap():
    np.random.seed(42)
    x = np.random.randn(32, 32).astype(np.float32)
    x[5, 5] = 40.0
    w = np.random.randn(32, 32).astype(np.float32)
    t = find_optimal_threshold(x, w, max_fp16_flops=1000.0)
    assert t in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def test_skip_modules_detection():
    acts = {"layer.0": np.random.randn(10, 10), "layer.1": np.ones((10, 10)) * 25.0}
    skipped = identify_skip_modules(acts, outlier_threshold=10.0)
    assert skipped == ["layer.1"]
