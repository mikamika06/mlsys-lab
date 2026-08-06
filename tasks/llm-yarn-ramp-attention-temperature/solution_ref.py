import math
import numpy as np


def yarn_ramp_temperature(
    q,
    k,
    inv_freq,
    positions,
    beta_fast,
    beta_slow,
    scale,
    temperature,
):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    num_freq = inv_freq.shape[0]
    freq = []
    denom_ramp = beta_fast - beta_slow
    for d in range(num_freq):
        val = (float(d) - beta_slow) / denom_ramp
        if val < 0.0:
            ramp = 0.0
        elif val > 1.0:
            ramp = 1.0
        else:
            ramp = val
        freq.append(float(inv_freq[d]) / (1.0 + ramp * (scale - 1.0)))

    n_q = q.shape[0]
    n_k = k.shape[0]
    dim = q.shape[1]

    qr = np.empty((n_q, dim), dtype=np.float64)
    for i in range(n_q):
        pos = positions[i]
        for j in range(num_freq):
            angle = pos * freq[j]
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = q[i, 2 * j]
            x1 = q[i, 2 * j + 1]
            qr[i, 2 * j] = x0 * c - x1 * s
            qr[i, 2 * j + 1] = x0 * s + x1 * c

    kr = np.empty((n_k, dim), dtype=np.float64)
    for i in range(n_k):
        pos = positions[i]
        for j in range(num_freq):
            angle = pos * freq[j]
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = k[i, 2 * j]
            x1 = k[i, 2 * j + 1]
            kr[i, 2 * j] = x0 * c - x1 * s
            kr[i, 2 * j + 1] = x0 * s + x1 * c

    scale_factor = math.sqrt(float(dim)) * math.sqrt(float(temperature))

    out = np.empty((n_q, n_k), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_k):
            acc = 0.0
            for d in range(dim):
                acc += qr[i, d] * kr[j, d]
            out[i, j] = acc / scale_factor

    return out
