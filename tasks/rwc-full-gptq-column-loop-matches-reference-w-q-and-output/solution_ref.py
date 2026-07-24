import numpy as np


def gptq_quantize(W, X, bits=3, group_size=2, damp=0.01):
    W_work = np.asarray(W, dtype=np.float64).copy()
    _, n = W_work.shape

    H = X @ X.T
    H = H + np.eye(n) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)

    W_q = np.zeros_like(W_work)
    maxq = (1 << (bits - 1)) - 1

    scales = {}
    for start in range(0, n, group_size):
        end = min(n, start + group_size)
        scale = np.max(np.abs(W_work[:, start:end]), axis=1) / maxq
        scale[scale == 0] = 1.0
        scales[start] = scale

    for i in range(n):
        start = (i // group_size) * group_size
        scale = scales[start]
        q = np.clip(np.round(W_work[:, i] / scale), -maxq, maxq) * scale
        W_q[:, i] = q
        err = q - W_work[:, i]
        if i + 1 < n:
            coeff = Hinv[i, i + 1:] / Hinv[i, i]
            W_work[:, i + 1:] -= np.outer(err, coeff)

    return W_q, W_q @ X
