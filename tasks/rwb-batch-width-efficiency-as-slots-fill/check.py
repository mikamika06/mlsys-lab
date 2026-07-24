import numpy as np

from mlsys import scorers


def _oracle(occupancy, N):
    per_step = np.asarray(occupancy, dtype=np.float64) / N
    mean = float(np.mean(per_step))
    return per_step, mean


def _cases():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(6):
        T = int(rng.integers(5, 200))
        N = int(rng.integers(1, 64))
        occupancy = rng.integers(0, N + 1, size=T)
        out.append((occupancy, N))
    # exact edge cases: always-full and always-empty batch
    out.append((np.full(10, 5, dtype=np.int64), 5))
    out.append((np.zeros(8, dtype=np.int64), 3))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for occupancy, N in _cases():
        ref_per_step, ref_mean = _oracle(occupancy, N)
        try:
            got = sol.batch_width_utilization(occupancy.copy(), N)
            got_per_step = np.asarray(got["per_step"], dtype=np.float64)
            got_mean = float(got["mean"])
        except Exception:
            return {"rel_err": float("inf")}

        if got_per_step.shape != ref_per_step.shape:
            return {"rel_err": float("inf")}
        if not np.all(np.isfinite(got_per_step)) or not np.isfinite(got_mean):
            return {"rel_err": float("inf")}

        ref_vec = np.concatenate([ref_per_step, [ref_mean]])
        got_vec = np.concatenate([got_per_step, [got_mean]])
        err = scorers.rel_err(ref_vec, got_vec)
        worst = max(worst, err)

    return {"rel_err": worst}
