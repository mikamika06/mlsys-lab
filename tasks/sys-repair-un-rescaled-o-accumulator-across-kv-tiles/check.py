import numpy as np
from mlsys import scorers


def _oracle_dense_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)

    cases = [
        (
            np.array([[4.0, 0.0], [3.5, 0.1]], dtype=np.float64),
            np.array(
                [[0.0, 1.0], [1.0, 0.0], [8.0, 8.0], [7.5, 7.5]],
                dtype=np.float64,
            ),
            np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 3.0], [-1.0, 4.0]], dtype=np.float64),
            2,
        ),
        (
            rng.normal(size=(5, 4)).astype(np.float64) * 2.0,
            rng.normal(size=(9, 4)).astype(np.float64) * 3.0,
            rng.normal(size=(9, 4)).astype(np.float64),
            3,
        ),
        (
            rng.normal(size=(6, 3)).astype(np.float64),
            rng.normal(size=(8, 3)).astype(np.float64) + 4.0,
            rng.normal(size=(8, 3)).astype(np.float64),
            1,
        ),
    ]

    worst = 0.0

    for Q, K, V, tile_size in cases:
        reference = _oracle_dense_attention(Q, K, V)
        try:
            candidate = sol.flash_attention_tiled(Q, K, V, tile_size)
            error = scorers.rel_err(reference, candidate)
        except Exception:
            return {"rel_err": float("inf")}

        if not np.isfinite(error):
            return {"rel_err": float("inf")}

        worst = max(worst, error)

    return {"rel_err": worst}
