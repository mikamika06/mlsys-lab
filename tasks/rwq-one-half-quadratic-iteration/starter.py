import math

def hqq_half_quadratic_step(W: list[list[float]], s: list[float], z: list[float], W_q: list[list[float]], lp: float, beta: float, qmin: int, qmax: int) -> tuple[list[list[float]], list[float]]:
    """
    Run one HQQ half-quadratic-splitting iteration (per-row zero-point
    groups), as described in task.md:
      1. residual = W_q - (W/s + z)
      2. shrink the residual with the generalized-Lp shrinkage operator
      3. update z as the row-mean of (W_q - shrunk_residual - W/s)
      4. re-quantize: W_q_new = clip(round(W/s + z_new), qmin, qmax)

    Returns (W_q_new, z_new).
    """
    raise NotImplementedError('your code here')
