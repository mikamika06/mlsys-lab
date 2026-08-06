import numpy as np
from mlsys import scorers

def _ref(logits, targets, mask):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = np.log(probs + 1e-12)
    idx = targets[..., None]
    log_target = np.take_along_axis(log_probs, idx, axis=-1).squeeze(-1)
    ce = -log_target
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        ce = ce * mask
        denom = np.sum(mask, axis=-1)
        loss = np.where(denom>0, np.sum(ce, axis=-1)/denom, 0.0)
    else:
        loss = np.mean(ce, axis=-1)
    return loss.astype(np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = []
    # case 1: small batch, mask None
    logits = rng.normal(size=(2,3,5))
    targets = rng.integers(low=0, high=5, size=(2,3))
    cases.append((logits, targets, None))
    # case 2: with mask
    logits = rng.normal(size=(4,6,10))
    targets = rng.integers(0,10,size=(4,6))
    mask = rng.choice([True, False], size=(4,6), p=[0.7,0.3])
    cases.append((logits, targets, mask))
    # case 3: all tokens masked out (edge)
    logits = rng.normal(size=(1,5,8))
    targets = rng.integers(0,8,size=(1,5))
    mask = np.zeros((1,5), dtype=bool)
    cases.append((logits, targets, mask))

    max_err = 0.0
    for logits, targets, mask in cases:
        logits_list = logits.tolist()
        targets_list = targets.tolist()
        mask_list = mask.tolist() if mask is not None else None
        try:
            got = sol.cross_entropy_loss(logits_list, targets_list, mask_list)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _ref(logits, targets, mask)
        err = scorers.max_abs_err(ref, np.array(got, dtype=np.float64))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
