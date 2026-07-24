import numpy as np


def _oracle(W, col_norms, keep_ratio):
    scores = np.abs(np.asarray(W, dtype=np.float64)) * np.asarray(col_norms, dtype=np.float64)[None, :]
    rows, cols = scores.shape
    k = max(1, int(round(cols * keep_ratio)))
    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        order = np.argsort(-scores[i], kind="stable")
        mask[i, order[:k]] = True
    return mask


def grade(sol, fx) -> dict:
    W = fx["wanda_w"]
    col_norms = fx["wanda_norms"]
    rows = W.shape[0]

    rng = np.random.default_rng(7)
    ratios = [0.3, 0.5, 0.7]

    for ratio in ratios:
        mask_ref = _oracle(W, col_norms, ratio)

        # baseline: identity scaling (c = 1) must reproduce the oracle mask
        scale_vecs = [np.ones(rows, dtype=np.float64)]
        # positive per-row scales spanning several orders of magnitude
        for _ in range(2):
            scale_vecs.append(rng.uniform(0.05, 20.0, size=rows))

        for c in scale_vecs:
            W_scaled = c[:, None] * W
            try:
                got = np.asarray(sol.wanda_mask(W_scaled.copy(), col_norms.copy(), ratio))
            except Exception:
                return {"exact_match": 0.0}

            if got.shape != mask_ref.shape:
                return {"exact_match": 0.0}

            if not np.array_equal(got.astype(bool), mask_ref):
                return {"exact_match": 0.0}

    return {"exact_match": 1.0}
