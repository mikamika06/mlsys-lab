import numpy as np
from mlsys import scorers, cachesim


def _ref_branchy(logits: np.ndarray, k: int) -> np.ndarray:
    """Reference branchy implementation."""
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.size
    mask = np.full_like(logits, -np.inf)
    # get threshold of top-k
    if k <= 0:
        return mask
    tau = np.partition(logits, -k)[-k]
    for i in range(n):
        if logits[i] >= tau:
            mask[i] = logits[i]
    return mask


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok1 = 1.0
    ok2 = 1.0

    # deterministic random cases
    cases = [rng.standard_normal(sz) * 3.0 for sz in [8, 17, 33, 64, 257]]

    for logits in cases:
        for k in [1, 3, 5, logits.size // 2]:
            ref_mask = _ref_branchy(logits, k)
            try:
                cand_mask = sol.branchless_topk_mask(logits.copy(), k)
            except Exception:
                return {"argmax_agreement": 0.0, "byte_exact_fraction": 0.0}

            a1 = scorers.argmax_agreement(ref_mask, cand_mask)
            b1 = scorers.byte_exact_fraction(ref_mask, cand_mask.tobytes())
            ok1 *= (1.0 if a1 == 1.0 else 0.0)
            ok2 *= (1.0 if b1 == 1.0 else 0.0)
            if not ok1 or not ok2:
                break
        if not ok1 or not ok2:
            break

    return {"argmax_agreement": ok1, "byte_exact_fraction": ok2}
