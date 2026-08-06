import math

def adam_trajectory(
    params0: list[float],
    grads: list[list[float]],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> list[list[float]]:
    """TODO: Implement Adam with bias correction.
This starter uses the *unbiased* moments directly, which leads to a biased trajectory."""
    raise NotImplementedError('your code here')
