import numpy as np


def obs_prune_step(H: np.ndarray, w: np.ndarray):
    """
    Compute the Optimal-Brain-Surgeon single-weight pruning step, as
    described in task.md:
      s_q     = w_q^2 / [H^-1]_qq            for every q
      q       = argmin_q s_q
      delta_w = -(w_q / [H^-1]_qq) * H^-1 @ e_q
      dL      = 0.5 * delta_w^T H delta_w

    Returns (q, delta_w, dL).
    """
    raise NotImplementedError('your code here')
