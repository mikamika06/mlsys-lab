import numpy as np

def lr_schedule(total_steps: int,
                warmup_steps: int,
                base_lr: float,
                min_lr: float = 0.0) -> np.ndarray:
    """
    Return a cosine learning‑rate schedule with optional linear warm‑up.

    Parameters
    ----------
    total_steps : int
        Total number of training steps.
    warmup_steps : int
        Number of initial steps over which the rate rises linearly from 0 to ``base_lr``.
    base_lr : float
        Peak learning rate reached after warm‑up.
    min_lr : float, default 0.0
        Minimum learning rate at the end of the cosine decay.

    Returns
    -------
    np.ndarray
        Array of shape (total_steps,) containing the learning rate for each step.
    """
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
