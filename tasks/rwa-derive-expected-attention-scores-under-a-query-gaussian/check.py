import numpy as np


def _oracle_expected_attention_scores(queries, keys, top_k):
    queries = np.asarray(queries, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    d = queries.shape[1]

    mu = np.mean(queries, axis=0)
    centered = queries - mu
    cov = centered.T @ centered / (queries.shape[0] - 1)

    mean_term = keys @ mu / np.sqrt(d)
    variance_term = np.einsum("ij,jk,ik->i", keys, cov, keys) / d
    scores = mean_term + 0.5 * variance_term
    order = np.argsort(-scores, kind="stable")[:top_k]
    return scores.astype(np.float64), order.astype(np.int64)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [1.0, 0.0, 2.0],
                [0.0, 2.0, 1.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, -1.0, 0.5],
            ]),
            2,
        ),
        (
            np.array([
                [3.0, -1.0],
                [2.0, 1.0],
                [0.0, 4.0],
                [-1.0, 2.0],
                [1.0, 0.0],
            ]),
            np.array([
                [2.0, 2.0],
                [-2.0, 1.0],
                [0.5, 3.0],
            ]),
            1,
        ),
        (
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, 1.0, 3.0],
                [0.0, 1.0, 2.0, 1.0],
                [3.0, 2.0, 0.0, -1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [2.0, 2.0, -1.0, 0.5],
            ]),
            2,
        ),
    ]

    rel_err = 0.0
    selected_exact = 1.0

    for queries, keys, top_k in cases:
        ref_scores, ref_idx = _oracle_expected_attention_scores(
            queries, keys, top_k
        )
        try:
            got_scores, got_idx = sol.expected_attention_scores(
                queries, keys, top_k
            )
            got_scores = np.asarray(got_scores, dtype=np.float64)
            got_idx = np.asarray(got_idx, dtype=np.int64)
        except Exception:
            return {"rel_err": float("inf"), "selected_exact": 0.0}

        denom = np.linalg.norm(ref_scores) + 1e-12
        rel_err = max(rel_err, float(np.linalg.norm(got_scores - ref_scores) / denom))
        if not np.array_equal(got_idx, ref_idx):
            selected_exact = 0.0

    return {
        "rel_err": rel_err,
        "selected_exact": selected_exact,
    }
