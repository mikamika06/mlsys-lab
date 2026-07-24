import numpy as np

def _reference_indices(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Brute‑force nearest neighbour indices with tie‑breaking by lowest index."""
    dists = np.linalg.norm(points[None, :, :] - queries[:, None, :], axis=2)
    return np.argmin(dists, axis=1)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    n, d = 50, 3
    m = 20
    points = rng.standard_normal((n, d))
    queries = rng.standard_normal((m, d))

    try:
        tree = sol.build_kd_tree(points)
        preds = []
        for q in queries:
            idx = sol.query_kd_tree(tree, q)
            if not isinstance(idx, int):
                return {"exact_match": 0.0}
            preds.append(idx)
        preds = np.array(preds, dtype=int)
    except Exception:
        return {"exact_match": 0.0}

    ref = _reference_indices(points, queries)
    ok = float(np.array_equal(preds, ref))
    return {"exact_match": ok}
