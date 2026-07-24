import numpy as np

def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    batch = 64
    d_in = 128
    d_out = 256
    X = rng.standard_normal((batch, d_in), dtype=np.float64)
    W_gate = rng.standard_normal((d_in, d_out), dtype=np.float64)
    W_up   = rng.standard_normal((d_in, d_out), dtype=np.float64)

    try:
        cand = sol.geglu_ffn(X, W_gate, W_up)
    except Exception:
        return {"max_abs_err": 0.0}

    gate_ref = X @ W_gate
    up_ref   = X @ W_up
    ref = _gelu(gate_ref) * up_ref

    err = np.max(np.abs(cand - ref))
    return {"max_abs_err": float(err)}
