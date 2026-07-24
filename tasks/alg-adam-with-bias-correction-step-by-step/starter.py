import numpy as np

def adam_trajectory(params0: np.ndarray, grads: np.ndarray, lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-08) -> np.ndarray:
    """TODO: Implement Adam with bias correction.
This starter uses the *unbiased* moments directly, which leads to a biased trajectory."""
    raise NotImplementedError('your code here')
