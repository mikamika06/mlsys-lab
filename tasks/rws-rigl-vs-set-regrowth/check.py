import numpy as np


def _oracle(w, g, mask, k, seed):
    mask = np.asarray(mask, dtype=bool)
    g = np.asarray(g, dtype=np.float64)
    shape = mask.shape

    flat_mask = mask.ravel()
    flat_abs_g = np.abs(g).ravel()
    zero_idx = np.flatnonzero(~flat_mask)

    order = np.argsort(-flat_abs_g[zero_idx], kind="stable")
    rigl_pick = zero_idx[order[:k]]
    rigl_mask = flat_mask.copy()
    rigl_mask[rigl_pick] = True

    rng = np.random.default_rng(seed)
    set_pick = rng.choice(zero_idx, size=k, replace=False)
    set_mask = flat_mask.copy()
    set_mask[set_pick] = True

    return rigl_mask.reshape(shape), set_mask.reshape(shape)


def _cases():
    rng = np.random.default_rng(9)
    cases = []

    shape = (5, 8)
    w = rng.normal(size=shape)
    g = rng.normal(size=shape) * 3.0
    mask = rng.random(shape) < 0.4
    cases.append((w, g, mask, 6, 123))

    shape2 = (10, 10)
    w2 = rng.normal(size=shape2)
    g2 = rng.normal(size=shape2)
    mask2 = rng.random(shape2) < 0.7
    n_zero2 = int((~mask2).sum())
    cases.append((w2, g2, mask2, min(4, n_zero2), 7))

    shape3 = (12,)
    w3 = rng.normal(size=shape3)
    # Deliberately include tied gradient magnitudes to exercise the
    # tie-break-by-lowest-index rule.
    g3 = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 0.0, 0.0, 1.5, -1.5, 2.0])
    mask3 = np.array([True, False, True, False, False, False, True, False, False, False, False, False])
    cases.append((w3, g3, mask3, 3, 42))

    return cases


def grade(sol, fx) -> dict:
    ok = 1.0
    for w, g, mask, k, seed in _cases():
        ref_rigl, ref_set = _oracle(w, g, mask, k, seed)

        try:
            got_rigl, got_set = sol.regrow_masks(
                np.array(w, copy=True), np.array(g, copy=True), np.array(mask, copy=True), k, seed
            )
        except Exception:
            return {"exact_match": 0.0}

        got_rigl = np.asarray(got_rigl, dtype=bool)
        got_set = np.asarray(got_set, dtype=bool)

        if got_rigl.shape != ref_rigl.shape or got_set.shape != ref_set.shape:
            return {"exact_match": 0.0}

        mask_b = np.asarray(mask, dtype=bool)
        # Every original active position must remain active.
        if not (np.all(got_rigl[mask_b]) and np.all(got_set[mask_b])):
            ok = 0.0
        # Exactly k new positions must have been reactivated.
        if int(got_rigl.sum()) != int(mask_b.sum()) + k:
            ok = 0.0
        if int(got_set.sum()) != int(mask_b.sum()) + k:
            ok = 0.0

        if not np.array_equal(got_rigl, ref_rigl):
            ok = 0.0
        if not np.array_equal(got_set, ref_set):
            ok = 0.0

    return {"exact_match": ok}
