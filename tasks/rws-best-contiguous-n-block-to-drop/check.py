import numpy as np


def _oracle(hidden_states, n):
    H = np.asarray(hidden_states, dtype=np.float64)
    B, L, _ = H.shape
    scores = []
    for s in range(L - n):
        a = H[:, s, :]
        b = H[:, s + n, :]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.sum(a * b, axis=1) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        scores.append(float(np.mean(np.arccos(cosine))))
    idx = int(np.argmin(np.asarray(scores)))
    return idx, scores[idx]


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.1, 0.99]],
                [[0.0, 1.0], [0.01, 0.99], [1.0, 0.0], [0.99, 0.1]],
            ]),
            1,
        ),
        (
            np.array([
                [[1, 0, 0], [0, 1, 0], [0.7, 0.7, 0], [0.9, 0.1, 0]],
                [[0, 1, 0], [0.7, 0.7, 0], [1, 0, 0], [0.8, 0.2, 0]],
            ], dtype=np.float64),
            2,
        ),
        (
            np.array([
                [[1, 2], [2, 1], [1.1, 2.1], [2.2, 1.1], [0.1, 1]],
                [[2, 1], [1, 2], [2.1, 1.1], [1.2, 2.2], [1, 0.1]],
                [[1.5, 1.5], [1.6, 1.4], [1.4, 1.6], [1.5, 1.4], [1.4, 1.5]],
            ], dtype=np.float64),
            1,
        ),
    ]

    index_ok = 1.0
    distance_ok = 1.0

    for H, n in cases:
        ref_idx, ref_dist = _oracle(H, n)
        try:
            got_idx, got_dist = sol.best_contiguous_n_block_to_drop(H, n)
        except Exception:
            return {"argmin_index": 0.0, "distance_error": float("inf")}

        if int(got_idx) != ref_idx:
            index_ok = 0.0
        if abs(float(got_dist) - ref_dist) > 1e-6:
            distance_ok = 0.0

    return {
        "argmin_index": index_ok,
        "distance_error": 0.0 if distance_ok else 1.0
    }
