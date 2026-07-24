import numpy as np


def _oracle(w, s):
    flat = np.asarray(w).reshape(-1)
    mags = np.abs(flat).astype(np.float64)
    threshold = float(np.quantile(mags, s))
    k = int(np.ceil((1.0 - s) * len(mags)))

    indices = np.arange(len(mags))
    order = np.lexsort((indices, -mags))
    keep = order[:k]

    mask = np.zeros(len(mags), dtype=bool)
    mask[keep] = True
    return threshold, mask.reshape(np.asarray(w).shape)


def grade(sol, fx) -> dict:
    cases = [
        (np.array([0.2, -0.9, 0.9, 0.1, 0.9]), 0.6),
        (np.array([1.0, -1.0, 1.0, 2.0, -2.0, 0.0]), 0.5),
        (np.array([[3.0, -3.0], [3.0, 1.0], [1.0, 0.0]]), 0.7),
        (np.array([5, 5, 5, 4, 4, 3, 2], dtype=np.float64), 0.4),
    ]

    for w, s in cases:
        try:
            got_threshold, got_mask = sol.quantile_keep_mask(w.copy(), s)
        except Exception:
            return {"exact_match": 0.0}

        ref_threshold, ref_mask = _oracle(w, s)

        if not np.isclose(float(got_threshold), ref_threshold, rtol=0.0, atol=1e-12):
            return {"exact_match": 0.0}

        if not np.array_equal(np.asarray(got_mask), ref_mask):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
