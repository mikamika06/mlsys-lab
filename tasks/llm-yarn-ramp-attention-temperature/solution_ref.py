import math


def yarn_ramp_temperature(
    q: list[list[float]],
    k: list[list[float]],
    inv_freq: list[float],
    positions: list[int],
    beta_fast: float,
    beta_slow: float,
    scale: float,
    temperature: float,
) -> list[list[float]]:
    num_freq = len(inv_freq)
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

    n_q = len(q)
    n_k = len(k)
    dim = len(q[0])

    qr = [[0.0] * dim for _ in range(n_q)]
    for i in range(n_q):
        pos = float(positions[i])
        for j in range(num_freq):
            angle = pos * freq[j]
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = q[i][2 * j]
            x1 = q[i][2 * j + 1]
            qr[i][2 * j] = x0 * c - x1 * s
            qr[i][2 * j + 1] = x0 * s + x1 * c

    kr = [[0.0] * dim for _ in range(n_k)]
    for i in range(n_k):
        pos = float(positions[i])
        for j in range(num_freq):
            angle = pos * freq[j]
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = k[i][2 * j]
            x1 = k[i][2 * j + 1]
            kr[i][2 * j] = x0 * c - x1 * s
            kr[i][2 * j + 1] = x0 * s + x1 * c

    scale_factor = math.sqrt(float(dim)) * math.sqrt(float(temperature))

    out = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            acc = 0.0
            for d in range(dim):
                acc += qr[i][d] * kr[j][d]
            out[i][j] = acc / scale_factor

    return out
