import numpy as np

def _layer_norm(x, gamma, beta, eps=1e-5):
    """Reference LayerNorm: normalise along the last axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, ddof=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def _ref_stack_forward(x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f,
                       n_blocks):
    """Pure-NumPy reference: N shared blocks + final norm."""
    h = x.copy()
    for _ in range(n_blocks):
        ln = _layer_norm(h, gamma, beta)
        mid = np.maximum(0.0, ln @ W1 + b1)       # ReLU
        out = mid @ W2 + b2
        h = h + out                                 # residual
    h = _layer_norm(h, gamma_f, beta_f)
    return h

def grade(sol, fx) -> dict:
    np.random.seed(42)
    cases = [
        (4, 8, 2, 3),
        (16, 32, 4, 5),
        (8, 16, 1, 10),
        (32, 64, 8, 2),
        (16, 16, 3, 7),
    ]
    max_err = 0.0
    for d, d_hidden, batch, n_blocks in cases:
        x       = np.random.randn(batch, d)
        gamma   = np.random.randn(d)
        beta    = np.random.randn(d) * 0.5
        W1      = np.random.randn(d, d_hidden) * 0.1
        b1      = np.random.randn(d_hidden) * 0.1
        W2      = np.random.randn(d_hidden, d) * 0.1
        b2      = np.random.randn(d) * 0.1
        gamma_f = np.random.randn(d)
        beta_f  = np.random.randn(d) * 0.5

        expected = _ref_stack_forward(x, gamma, beta, W1, b1, W2, b2,
                                      gamma_f, beta_f, n_blocks)
        try:
            got = sol.stack_blocks_forward(
                x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f, n_blocks
            )
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - expected)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
