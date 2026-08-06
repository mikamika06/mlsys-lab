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

    d = H.shape[0]
    M = []
    for i in range(d):
        row = []
        for j in range(d):
            row.append(float(H[i, j]))
        for j in range(d):
            row.append(1.0 if i == j else 0.0)
        M.append(row)

    for i in range(d):
        max_val = abs(M[i][i])
        max_row = i
        for k in range(i + 1, d):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        for j in range(2 * d):
            M[i][j] /= pivot

        for k in range(d):
            if k != i:
                factor = M[k][i]
                for j in range(2 * d):
                    M[k][j] -= factor * M[i][j]

    Hinv = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            Hinv[i, j] = M[i][d + j]

    diag = np.zeros(d, dtype=np.float64)
    for i in range(d):
        diag[i] = Hinv[i, i]

    s = np.zeros(d, dtype=np.float64)
    for i in range(d):
        s[i] = (w[i] ** 2) / diag[i]

    q = 0
    min_val = s[0]
    for i in range(1, d):
        if s[i] < min_val:
            min_val = s[i]
            q = i

    c = w[q] / diag[q]
    delta_w = np.zeros(d, dtype=np.float64)
    for i in range(d):
        delta_w[i] = -c * Hinv[i, q]

    u = np.zeros(d, dtype=np.float64)
    for j in range(d):
        acc = 0.0
        for i in range(d):
            acc += delta_w[i] * H[i, j]
        u[j] = acc

    quad_form = 0.0
    for j in range(d):
        quad_form += u[j] * delta_w[j]
    dL = 0.5 * quad_form

    return q, delta_w, dL
