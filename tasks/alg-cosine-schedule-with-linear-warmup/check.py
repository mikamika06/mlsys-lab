import numpy as np
from mlsys.scorers import max_abs_err

def _reference(total_steps, warmup_steps, base_lr, min_lr):
    steps = np.arange(total_steps, dtype=np.int64)
    lrs = np.empty(total_steps, dtype=np.float64)

    # Linear warm‑up
    if warmup_steps > 0:
        mask_warm = steps < warmup_steps
        t_w = steps[mask_warm]
        lrs[mask_warm] = base_lr * (t_w + 1) / warmup_steps

    # Cosine decay after warm‑up
    if total_steps > warmup_steps:
        mask_cos = steps >= warmup_steps
        t_c = steps[mask_cos] - warmup_steps
        T = total_steps - warmup_steps
        lrs[mask_cos] = min_lr + (base_lr - min_lr) * (
            1 + np.cos(np.pi * t_c / T)
        ) / 2

    return lrs

def grade(sol, fx) -> dict:
    cases = [
        (5, 2, 0.1, 0.01),
        (10, 0, 0.05, 0.005),
        (20, 5, 0.2, 0.02),
        (7, 3, 0.15, 0.015)
    ]

    max_err = 0.0
    for total_steps, warmup_steps, base_lr, min_lr in cases:
        try:
            got = sol.lr_schedule(total_steps, warmup_steps, base_lr, min_lr)
            ref = _reference(total_steps, warmup_steps, base_lr, min_lr)
            if got.shape != ref.shape:
                return {"max_abs_err": 1.0}
            err = max_abs_err(ref, got)
        except Exception:
            return {"max_abs_err": 1.0}
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
