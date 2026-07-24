import numpy as np

def _silu(x):
    return x / (1.0 + np.exp(-x))

def grade(sol, fx) -> dict:
    # Generate a handful of random test cases
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        n = rng.integers(2, 10)
        d_in = rng.integers(3, 8)
        d_out = rng.integers(2, 6)

        X = rng.standard_normal((n, d_in))
        W_gate = rng.standard_normal((d_in, d_out))
        W_up   = rng.standard_normal((d_in, d_out))

        # Randomly decide whether to include biases
        if rng.random() < 0.5:
            b_gate = None
        else:
            b_gate = rng.standard_normal(d_out)
        if rng.random() < 0.5:
            b_up = None
        else:
            b_up = rng.standard_normal(d_out)

        # Reference implementation
        gate_ref = X @ W_gate + (b_gate if b_gate is not None else 0.0)
        up_ref   = X @ W_up   + (b_up   if b_up   is not None else 0.0)
        ref = gate_ref * _silu(up_ref)

        # Candidate implementation
        try:
            cand = sol.swiglu(X, W_gate, W_up, b_gate, b_up)
        except Exception as e:
            return {"max_abs_err": float("inf")}

        err = np.max(np.abs(cand - ref))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
