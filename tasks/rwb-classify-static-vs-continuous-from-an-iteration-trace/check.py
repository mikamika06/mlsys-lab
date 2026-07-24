import numpy as np


def _oracle_label(active_2d: np.ndarray) -> str:
    active_2d = np.asarray(active_2d).astype(bool)
    T = active_2d.shape[0]
    for t in range(1, T):
        prev = active_2d[t - 1]
        curr = active_2d[t]
        new_ids = curr & ~prev
        continuing = prev & curr
        if new_ids.any() and continuing.any():
            return "continuous"
    return "static"


def grade(sol, fx) -> dict:
    active = np.asarray(fx["active"])
    run_len = np.asarray(fx["run_len"])

    ok = 1.0
    for r in range(active.shape[0]):
        T = int(run_len[r])
        run = active[r, :T, :]
        expected = _oracle_label(run)
        try:
            got = sol.classify_scheduling(run.copy())
        except Exception:
            ok = 0.0
            break
        if str(got).strip().lower() != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
