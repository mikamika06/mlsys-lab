import numpy as np
import math

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
    lrs = np.empty(total_steps, dtype=np.float64)

    for step in range(total_steps):
        if warmup_steps > 0 and step < warmup_steps:
            lrs[step] = base_lr * (step + 1) / warmup_steps
        elif total_steps > warmup_steps:
            t_c = step - warmup_steps
            T = total_steps - warmup_steps
            lrs[step] = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * t_c / T)) / 2

    return lrs
