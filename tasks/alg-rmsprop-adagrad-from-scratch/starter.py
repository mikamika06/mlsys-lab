import numpy as np

def rmsprop_trajectory(grads: np.ndarray, lr: float=0.01, eps: float=1e-08, decay_rate: float=0.9) -> np.ndarray:
    """TODO: This implementation uses plain SGD instead of RMSProp.
It will fail the max_abs_err gate because it does not adapt the step size."""
    raise NotImplementedError('your code here')
