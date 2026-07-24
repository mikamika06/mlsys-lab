import numpy as np
from mlsys import scorers

def _stable_cross_entropy(logits, target):
    """Numerically stable cross-entropy via the log-sum-exp trick.

    Oracle used as the reference for comparison. The solution_ref.py
    implements the same algorithm so that grade(sol, sol) yields
    rel_err == 0.
    """
    m = np.max(logits)
    log_sum_exp = m + np.log(np.sum(np.exp(logits - m)))
    return float(-(logits[target] - log_sum_exp))

def grade(sol, fx) -> dict:
    """Return ``{"rel_err": …}`` comparing candidate outputs to the oracle.

    For acceptance, ``grade(solution_ref, solution_ref)`` must pass,
    so we always evaluate ``sol.cross_entropy`` and compare against
    the internal stable oracle.
    """
    rng = np.random.RandomState(42)

    refs = []
    cands = []

    # --- moderate random cases ---
    for _ in range(20):
        C = int(rng.randint(2, 10))
        logits = rng.randn(C).astype(np.float64) * 1000.0
        target = int(rng.randint(C))

        ref_val = _stable_cross_entropy(logits, target)
        try:
            cand_val = float(sol.cross_entropy(logits.copy(), target))
        except Exception:
            cand_val = float("nan")

        refs.append(ref_val)
        cands.append(cand_val)

    # --- extreme-magnitude cases that break naive implementations ---
    for scale in [1e-3, 1e2, 1e5, -1e5, 1e6]:
        C = 5
        logits = np.full(C, scale, dtype=np.float64)
        for t in range(C):
            ref_val = _stable_cross_entropy(logits, t)
            try:
                cand_val = float(sol.cross_entropy(logits.copy(), t))
            except Exception:
                cand_val = float("nan")
            refs.append(ref_val)
            cands.append(cand_val)

    ref_arr = np.asarray(refs, dtype=np.float64)
    cand_arr = np.asarray(cands, dtype=np.float64)

    rel_err = scorers.rel_err(ref_arr, cand_arr)
    return {"rel_err": float(rel_err)}
