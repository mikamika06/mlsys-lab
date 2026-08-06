import math

def compute_dx(dy: list[list[float]],
               x: list[list[float]],
               gamma: list[float],
               beta: list[float],
               eps: float = 1e-5) -> list[list[float]]:
    """TODO: This implementation is incorrect. It ignores the second term in the
analytic gradient that involves `x_hat` and the mean of `dy * x_hat`.
As a result, the returned gradients will have a relative error far above
the required threshold."""
    raise NotImplementedError('your code here')
