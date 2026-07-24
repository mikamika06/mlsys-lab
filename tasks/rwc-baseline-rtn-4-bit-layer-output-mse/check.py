import numpy as np
from mlsys.scorers import mse

def _quantize(W, group_size=16):
    out, in_ = W.shape
    padded_in = ((in_ + group_size - 1) // group_size) * group_size
    W_pad = np.pad(W, ((0, 0), (0, padded_in - in_)), mode="constant")
    groups = W_pad.reshape(out, padded_in // group_size, group_size)
    max_abs = np.max(np.abs(groups), axis=2, keepdims=True)
    scale = max_abs / 7.0
    # avoid division by zero
    scale[scale == 0] = 1.0
    q = np.round(groups / scale).astype(np.int8)
    q = np.clip(q, -8, 7)
    deq = q * scale
    W_hat = deq.reshape(out, padded_in)[:, :in_]
    return W_hat

def _oracle(W, X):
    W_hat = _quantize(W)
    return float(np.mean((W @ X - W_hat @ X) ** 2))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((32, 64)).astype(np.float64)
    X = rng.standard_normal((64, 128)).astype(np.float64)
    try:
        student_mse = sol.quantize_layer_output_mse(W, X)
    except Exception:
        return {"mse": float("inf")}
    oracle_mse = _oracle(W, X)
    diff = abs(student_mse - oracle_mse)
    return {"mse": diff}
