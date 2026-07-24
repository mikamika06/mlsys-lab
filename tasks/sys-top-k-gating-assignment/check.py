import numpy as np


def _oracle(logits, k):
    logits = np.asarray(logits, dtype=np.float64)
    N, E = logits.shape
    order = np.argsort(-logits, axis=1, kind="stable")
    indices = order[:, :k]
    rows = np.arange(N)[:, None]
    top_logits = logits[rows, indices]
    m = np.max(top_logits, axis=1, keepdims=True)
    e = np.exp(top_logits - m)
    weights = e / np.sum(e, axis=1, keepdims=True)
    return indices.astype(np.int64), weights


def _make_case(rng, N, E, k, with_ties):
    logits = rng.standard_normal((N, E))
    if with_ties:
        # force a tie between experts 0 and 1 on every row
        logits[:, 1] = logits[:, 0]
    return logits


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (6, 8, 2, False),
        (4, 5, 3, False),
        (5, 6, 1, False),
        (5, 4, 2, True),
    ]

    idx_ok = 1.0
    worst_rel_err = 0.0

    for N, E, k, with_ties in cases:
        logits = _make_case(rng, N, E, k, with_ties)
        ref_idx, ref_w = _oracle(logits, k)

        try:
            got_idx, got_w = sol.topk_gating(logits.copy(), k)
        except Exception:
            return {"exact_match": 0.0, "rel_err": float("inf")}

        try:
            got_idx = np.asarray(got_idx)
            got_w = np.asarray(got_w, dtype=np.float64)
        except Exception:
            return {"exact_match": 0.0, "rel_err": float("inf")}

        if got_idx.shape != ref_idx.shape or got_w.shape != ref_w.shape:
            return {"exact_match": 0.0, "rel_err": float("inf")}

        if not np.array_equal(got_idx, ref_idx):
            idx_ok = 0.0

        if not np.all(np.isfinite(got_w)):
            return {"exact_match": idx_ok, "rel_err": float("inf")}

        err = float(np.linalg.norm(got_w - ref_w) / (np.linalg.norm(ref_w) + 1e-12))
        worst_rel_err = max(worst_rel_err, err)

    return {"exact_match": idx_ok, "rel_err": worst_rel_err}
