import numpy as np

def vjp_mul_exp_log(x: np.ndarray, y: np.ndarray, upstream: np.ndarray):
    """
    Correct VJP for h(x,y)=log(exp(x*y)).

    Since log(exp(z)) = z, the Jacobian is diagonal with entries y (for x)
    and x (for y).  The vector-Jacobian product therefore reduces to
    elementwise multiplication of the upstream gradient with the other input.
    """
    grad_x = upstream * y
    grad_y = upstream * x
    return grad_x.astype(np.float64), grad_y.astype(np.float64)
