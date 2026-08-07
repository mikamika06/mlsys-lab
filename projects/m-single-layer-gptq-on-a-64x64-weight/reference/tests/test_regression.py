import sys
import numpy as np

sys.path.insert(0, ".")
from gptq_single.gptq import gptq_quantize, rtn_quantize

def test_gptq_beats_rtn_mse():
    rng = np.random.default_rng(123)
    W = rng.normal(size=(64, 64))
    X = rng.normal(size=(128, 64))
    W_gptq = gptq_quantize(W, X)
    W_rtn = rtn_quantize(W)
    out_orig = X @ W
    out_gptq = X @ W_gptq
    out_rtn = X @ W_rtn
    mse_gptq = np.mean((out_orig - out_gptq) ** 2)
    mse_rtn = np.mean((out_orig - out_rtn) ** 2)
    assert mse_gptq < mse_rtn, f"GPTQ MSE {mse_gptq} not less than RTN MSE {mse_rtn}"

def test_output_shape_preserved():
    rng = np.random.default_rng(456)
    W = rng.normal(size=(64, 64))
    X = rng.normal(size=(32, 64))
    W_q = gptq_quantize(W, X)
    assert W_q.shape == W.shape

def test_no_nan_or_inf():
    rng = np.random.default_rng(789)
    W = rng.normal(size=(64, 64))
    X = rng.normal(size=(64, 64))
    W_q = gptq_quantize(W, X)
    assert not np.isnan(W_q).any()
    assert not np.isinf(W_q).any()
