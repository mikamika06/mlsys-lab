import numpy as np


def offloaded_adamw_step(
    param,
    grad,
    m,
    v,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
):
    """Perform one CPU-offloaded AdamW optimizer step."""
    raise NotImplementedError("your code here")
