import numpy as np

def compute_dx(dy: np.ndarray, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float=1e-05) -> np.ndarray:
    """TODO: This implementation is incorrect. It ignores the second term in the
analytic gradient that involves `x_hat` and the mean of `dy * x_hat`.
As a result, the returned gradients will have a relative error far above
the required threshold."""
    raise NotImplementedError('your code here')
