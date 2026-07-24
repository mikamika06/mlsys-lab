import numpy as np
from mlsys.scorers import rel_err


def _oracle(lens, batch_ids):
    lens = np.asarray(lens, dtype=np.float64)
    batch_ids = np.asarray(batch_ids)
    total_slots = 0.0
    total_wasted = 0.0
    for b in np.unique(batch_ids):
        batch_lens = lens[batch_ids == b]
        max_len = float(np.max(batch_lens))
        batch_size = float(batch_lens.shape[0])
        slots = max_len * batch_size
        total_slots += slots
        total_wasted += slots - float(np.sum(batch_lens))
    return total_wasted / total_slots


def grade(sol, fx) -> dict:
    lens = np.asarray(fx["lens"], dtype=np.int64)
    batch_ids = np.asarray(fx["batch_ids"], dtype=np.int64)

    ref = _oracle(lens, batch_ids)

    try:
        got = float(sol.padding_waste_fraction(lens.copy(), batch_ids.copy()))
    except Exception:
        return {"rel_err": float("inf")}

    if not np.isfinite(got):
        return {"rel_err": float("inf")}

    return {"rel_err": rel_err(np.array([ref]), np.array([got]))}
