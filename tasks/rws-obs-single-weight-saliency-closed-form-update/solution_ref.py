import numpy as np


def obs_prune_step(H: np.ndarray, w: np.ndarray):
    """
    Optimal Brain Surgeon, single-weight pruning step.

    For a network at a local loss minimum (gradient ~ 0), the
    second-order loss increase from forcing w_q to exactly 0, minimized
    over how the OTHER weights compensate, has the closed form:

        s_q      = w_q^2 / [H^-1]_qq                       (saliency)
        delta_w  = -(w_q / [H^-1]_qq) * H^-1 @ e_q          (optimal update)
        dL       = 0.5 * delta_w^T H delta_w  ( == 0.5 * s_q at the argmin )

    Picks q = argmin_q s_q (least-damaging weight to prune).

    Returns (q, delta_w, dL):
      q       -- int, the chosen index.
      delta_w -- (d,) float array, the closed-form update (w[q]+delta_w[q] == 0).
      dL      -- float, the analytic second-order loss change.
    """
    H = np.asarray(H, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    Hinv = np.linalg.inv(H)
    diag = np.diag(Hinv)
    s = w ** 2 / diag
    q = int(np.argmin(s))

    c = w[q] / diag[q]
    delta_w = -c * Hinv[:, q]
    dL = 0.5 * float(delta_w @ H @ delta_w)
    return q, delta_w, dL
