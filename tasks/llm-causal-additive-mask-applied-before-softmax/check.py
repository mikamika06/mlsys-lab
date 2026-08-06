import numpy as np
from mlsys.scorers import max_abs_err

def _reference(scores):
    mask = np.tril(np.ones_like(scores, dtype=bool))
    masked_scores = scores.copy()
    masked_scores[~mask] = -np.inf
    exp_scores = np.exp(masked_scores)
    row_sums = exp_scores.sum(axis=-1, keepdims=True)
    return exp_scores / row_sums

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        np.array([[0, 1], [2, 3]], dtype=np.float64),
        rng.random((4, 4)),
        np.arange(9).reshape(3, 3).astype(np.float64),
        np.zeros((5, 5), dtype=np.float64),
        np.full((3, 3), -10.0, dtype=np.float64)
    ]
    max_err = 0.0
    for scores_arr in cases:
        scores_list = scores_arr.tolist()
        try:
            out_list = sol.causal_masked_softmax(scores_list)
        except Exception:
            return {"max_abs_err": float("inf")}

        out = np.array(out_list, dtype=np.float64)
        if out.shape != scores_arr.shape:
            return {"max_abs_err": float("inf")}

        ref = _reference(scores_arr)
        err = max_abs_err(ref, out)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
