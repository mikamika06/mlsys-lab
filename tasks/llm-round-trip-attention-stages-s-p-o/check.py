import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    n, d = 7, 5
    Q = rng.standard_normal((n, d))
    K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))

    # Reference implementation
    scores_ref = (Q @ K.T) / np.sqrt(d)
    exp_scores = np.exp(scores_ref - np.max(scores_ref, axis=-1, keepdims=True))
    probs_ref = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out_ref = probs_ref @ V

    try:
        S, P, O = sol.attention_roundtrip(Q, K, V)
    except Exception:
        return {"S": float("inf"), "P": float("inf"), "O": float("inf")}

    err_S = scorers.max_abs_err(S, scores_ref)
    err_P = scorers.max_abs_err(P, probs_ref)
    err_O = scorers.max_abs_err(O, out_ref)

    return {"S": err_S, "P": err_P, "O": err_O}
