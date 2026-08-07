import numpy as np
from q4k.quant import dequantize_q4_k, quantize_q4_k
from q4k.analysis import find_dominating_subblock
from q4k.compare import compare_q4k_q40_error


def test_round_trip_exact_bytes():
    rng = np.random.default.rng(42)
    w = rng.uniform(-2.0, 2.0, 256).astype(np.float32)
    data = quantize_q4_k(w)
    recon = dequantize_q4_k(data)
    data2 = quantize_q4_k(recon)
    assert data == data2, "round-trip quantized bytes do not match"


def test_dominating_subblock_bounds():
    rng = np.random.default.rng(43)
    w = rng.uniform(-1.0, 1.0, 256).astype(np.float32)
    sb_idx = find_dominating_subblock(w)
    assert 0 <= sb_idx < 8, "subblock index out of bounds"


def test_compare_returns_dict():
    rng = np.random.default.rng(44)
    w = rng.uniform(-1.5, 1.5, 256).astype(np.float32)
    res = compare_q4k_q40_error(w)
    assert isinstance(res, dict)
    assert "q4_0_mse" in res
    assert "q4_k_mse" in res
