import numpy as np

from mlsys import scorers


def _oracle_weights(chunk_scores: np.ndarray) -> np.ndarray:
    """Ground truth: plain global softmax over every concatenated raw
    score, computed stably in fp64 -- deliberately NOT going through the
    per-chunk LSE machinery, so it's an independent check on whatever
    reconstruction method the student used.
    """
    scores = np.asarray(chunk_scores, dtype=np.float64).reshape(-1)
    m = scores.max()
    e = np.exp(scores - m)
    return e / e.sum()


def _make_case(rng, C, chunk_size, d):
    chunk_scores = rng.normal(0.0, 4.0, size=(C, chunk_size))
    V = rng.normal(0.0, 1.0, size=(C, chunk_size, d))
    m = chunk_scores.max(axis=1, keepdims=True)
    chunk_lse = m[:, 0] + np.log(np.exp(chunk_scores - m).sum(axis=1))
    chunk_partial_out = (np.exp(chunk_scores - m)[:, :, None] * V).sum(axis=1)
    return chunk_scores, chunk_lse, chunk_partial_out


def _synthetic_cases():
    rng = np.random.default_rng(31)
    cases = []
    for C, chunk_size, d in [(3, 5, 4), (8, 2, 3), (4, 4, 8), (10, 3, 2)]:
        cases.append(_make_case(rng, C, chunk_size, d))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["chunk_scores"], fx["chunk_lse"], fx["chunk_partial_out"])] + _synthetic_cases()

    worst = 0.0
    for chunk_scores, chunk_lse, chunk_partial_out in cases:
        chunk_scores = np.asarray(chunk_scores, dtype=np.float64)
        chunk_lse = np.asarray(chunk_lse, dtype=np.float64)
        chunk_partial_out = np.asarray(chunk_partial_out, dtype=np.float64)

        ref = _oracle_weights(chunk_scores)

        try:
            got = sol.reconstruct_global_weights(
                chunk_scores.copy(), chunk_lse.copy(), chunk_partial_out.copy()
            )
            got = np.asarray(got, dtype=np.float64).reshape(-1)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(ref, got)
        worst = max(worst, err)

    return {"rel_err": worst}
