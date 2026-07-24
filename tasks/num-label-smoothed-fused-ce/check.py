import numpy as np
from mlsys.scorers import rel_err

def _ref(logits, targets, eps):
    """Reference label-smoothed fused CE built entirely in NumPy."""
    N, K = logits.shape
    # --- smoothed target matrix ---
    q_smooth = np.full((N, K), eps / K, dtype=np.float64)
    q_smooth[np.arange(N), targets] += 1.0 - eps
    # --- fused stable log-softmax ---
    m = np.max(logits, axis=1, keepdims=True)                       # (N, 1)
    log_Z = np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))  # (N, 1)
    log_p = logits - m - log_Z                                      # (N, K)
    # --- cross-entropy ---
    losses = -np.sum(q_smooth * log_p, axis=1)                      # (N,)
    return float(np.mean(losses))

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    cases = [
        # (N, K, eps)
        (5,  10,  0.1),
        (1,   3,  0.0),    # no smoothing — plain CE
        (1,   3,  1.0),    # full smoothing — uniform target
        (8, 100,  0.2),
        (32, 50,  0.1),
        (16, 200, 0.05),
    ]
    worst = 0.0
    for N, K, eps in cases:
        logits  = rng.randn(N, K).astype(np.float64) * 5.0
        targets = rng.randint(0, K, size=N)
        try:
            got = float(sol.label_smoothed_fused_ce(logits, targets, eps))
        except Exception:
            return {"rel_err": 1.0}
        expected = _ref(logits, targets, eps)
        err = rel_err(np.array([expected]), np.array([got]))
        worst = max(worst, err)
    return {"rel_err": worst}
