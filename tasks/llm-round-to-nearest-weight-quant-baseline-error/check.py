import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    # Fixed random tensor for reproducibility
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 7)).astype(np.float32)
    num_bits = 8

    try:
        Q = sol.round_to_nearest(W, num_bits)
    except Exception as e:
        return {"rel_err": float("inf")}

    # Reference implementation (oracle) inside the grader
    max_val = np.max(np.abs(W))
    scale = max_val / ((2**(num_bits-1))-1)
    qmin = -(2**(num_bits-1))
    qmax = 2**(num_bits-1)-1

    # De‑quantize candidate output
    try:
        W_hat = Q.astype(np.float32) * scale
    except Exception:
        return {"rel_err": float("inf")}

    # Compute relative error
    err = rel_err(W, W_hat)

    # Ensure dtype correctness: int8 for <=8 bits, else int16
    expected_dtype = np.int8 if num_bits <= 8 else np.int16
    if Q.dtype != expected_dtype:
        err += 1.0  # penalize wrong dtype heavily

    return {"rel_err": err}
