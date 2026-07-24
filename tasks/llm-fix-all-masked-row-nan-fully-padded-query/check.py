import numpy as np
from mlsys.scorers import max_abs_err


def _ref(scores, mask):
    """Independent oracle: per-row softmax over kept keys; zeros for a fully
    masked (fully padded) row. Loops over rows explicitly so no 0/0 or
    inf - inf is ever formed."""
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n, m = scores.shape
    out = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        keep = np.flatnonzero(mask[i])
        if keep.size == 0:
            continue  # fully padded query -> all zeros
        s = scores[i, keep]
        s = s - np.max(s)
        e = np.exp(s)
        out[i, keep] = e / e.sum()
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = []
    # 1. tiny hand case: one normal row, one fully padded row
    cases.append((
        np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64),
        np.array([[True, False], [False, False]], dtype=bool),
    ))
    # 2. every row fully padded
    cases.append((
        rng.standard_normal((3, 4)),
        np.zeros((3, 4), dtype=bool),
    ))
    # 3. random logits, one fully padded row inserted among valid rows
    s = rng.standard_normal((5, 6))
    k = rng.random((5, 6)) > 0.4
    k[0] = True                # a dense row
    k[2] = False               # a fully padded row
    k[4, 3] = True             # guarantee row 4 keeps something
    cases.append((s, k))
    # 4. large-magnitude logits (would overflow without the max-shift) + padded row
    s = rng.standard_normal((4, 5)) * 300.0
    k = rng.random((4, 5)) > 0.3
    k[1] = False               # fully padded
    k[3] = True                # dense
    cases.append((s, k))
    # 5. single kept key per row, plus a fully padded row
    s = rng.standard_normal((3, 3))
    k = np.array([[True, False, False],
                  [False, False, False],
                  [False, True, False]], dtype=bool)
    cases.append((s, k))

    max_err = 0.0
    for scores, mask in cases:
        scores = np.ascontiguousarray(scores, dtype=np.float64)
        mask = np.ascontiguousarray(mask, dtype=bool)
        try:
            out = sol.masked_softmax(scores, mask)
        except Exception:
            return {"max_abs_err": float("inf")}
        out = np.asarray(out)
        if out.shape != scores.shape:
            return {"max_abs_err": float("inf")}
        if not np.all(np.isfinite(out)):
            return {"max_abs_err": float("inf")}
        ref = _ref(scores, mask)
        err = max_abs_err(ref, out.astype(np.float64))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
