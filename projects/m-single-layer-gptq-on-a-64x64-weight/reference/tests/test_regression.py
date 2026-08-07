import sys
import numpy as np

sys.path.insert(0, ".")
from gptq.single import run_gptq_single
from gptq.compare import compare_mse


def test_gptq_shape():
    W = np.random.standard_normal((64, 64))
    X = np.random.standard_normal((128, 64))
    H = X.T @ X + 1e-4 * np.eye(64)
    invH = np.linalg.inv(H)
    W_q = run_gptq_single(W, invH, bits=4)
    assert W_q.shape == (64, 64)


def test_gptq_beats_rtn_mse():
    W = np.random.standard_normal((64, 64))
    X = np.random.standard_normal((128, 64))
    H = X.T @ X + 1e-4 * np.eye(64)
    invH = np.linalg.inv(H)
    mse_gptq, mse_rtn = compare_mse(W, X, invH, bits=4)
    assert mse_gptq < mse_rtn, f"GPTQ MSE {mse_gptq} not lower than RTN MSE {mse_rtn}"
