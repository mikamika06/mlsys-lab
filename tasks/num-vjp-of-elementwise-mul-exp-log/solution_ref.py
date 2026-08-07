def vjp_mul_exp_log(x: list[float], y: list[float], upstream: list[float]) -> tuple[list[float], list[float]]:
    """
    Correct VJP for h(x,y)=log(exp(x*y)).

    Since log(exp(z)) = z, the Jacobian is diagonal with entries y (for x)
    and x (for y).  The vector-Jacobian product therefore reduces to
    elementwise multiplication of the upstream gradient with the other input.
    """
    grad_x = [u * yi for u, yi in zip(upstream, y)]
    grad_y = [u * xi for u, xi in zip(upstream, x)]
    return grad_x, grad_y
