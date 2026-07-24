import numpy as np


def _oracle(W, X, sparsities):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    col_norm = np.linalg.norm(X, axis=0)
    S = np.abs(W) * col_norm[None, :]

    out_features, in_features = W.shape
    masks = []
    for s in sparsities:
        n_prune = int(round(s * in_features))
        mask = np.ones_like(W, dtype=np.int64)
        for o in range(out_features):
            order = np.argsort(S[o], kind="stable")
            prune_idx = order[:n_prune]
            mask[o, prune_idx] = 0
        masks.append(mask)
    return masks


def _build_cases():
    cases = []
    for seed, out_f, in_f, sparsities in [
        (0, 6, 20, [0.3, 0.5, 0.7]),
        (1, 4, 12, [0.25, 0.6]),
        (2, 8, 16, [0.1, 0.9]),
    ]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((out_f, in_f))
        X = rng.standard_normal((30, in_f)) * rng.uniform(0.1, 5.0, size=in_f)
        cases.append((W, X, sparsities))
    return cases


def grade(sol, fx) -> dict:
    total = 0
    matched = 0

    for W, X, sparsities in _build_cases():
        ref_masks = _oracle(W, X, sparsities)

        try:
            got_masks = sol.wanda_masks_for_sparsities(W.copy(), X.copy(), list(sparsities))
        except Exception:
            return {"exact_match": 0.0}

        if not isinstance(got_masks, (list, tuple)) or len(got_masks) != len(ref_masks):
            return {"exact_match": 0.0}

        for ref_m, got_m in zip(ref_masks, got_masks):
            got_m = np.asarray(got_m)
            if got_m.shape != ref_m.shape:
                return {"exact_match": 0.0}
            total += ref_m.size
            matched += int(np.sum(got_m.astype(np.int64) == ref_m))

    return {"exact_match": (matched / total) if total else 0.0}
