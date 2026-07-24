import numpy as np

def _max_abs_err(a, b):
    """Max absolute element-wise difference between two arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))

def _gelu(z):
    """GELU tanh approximation (PyTorch default)."""
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))

def _ffn_ref(x, W_up, b_up, W_down, b_down):
    """Reference FFN forward pass: down(gelu(up(x)))."""
    h = W_up @ x + b_up
    a = _gelu(h)
    return W_down @ a + b_down

def grade(sol, fx) -> dict:
    """Grade the learner's ffn_forward against a NumPy oracle reference."""
    worst = 0.0

    # Build test cases with fixed seeds so grading is deterministic.
    seeds_and_sizes = [(0, 8, 32), (100, 16, 64), (200, 4, 4),
                       (300, 32, 128), (42, 8, 8)]

    for seed, d, d_hidden in seeds_and_sizes:
        rng = np.random.RandomState(seed)
        W_up   = rng.randn(d_hidden, d) * 0.5
        b_up   = rng.randn(d_hidden) * 0.05
        W_down = rng.randn(d, d_hidden) * 0.5
        b_down = rng.randn(d) * 0.05
        x      = rng.randn(d) * 2.0

        ref = _ffn_ref(x, W_up, b_up, W_down, b_down)
        try:
            got = sol.ffn_forward(x, W_up, b_up, W_down, b_down)
        except Exception:
            return {"max_abs_err": float("inf")}
        err = _max_abs_err(ref, got)
        worst = max(worst, err)

    return {"max_abs_err": worst}
