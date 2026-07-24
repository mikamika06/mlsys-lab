import numpy as np

f16 = np.float16


def _welford_mean_var(x16):
    n = 0
    mean = f16(0.0)
    M2 = f16(0.0)
    for xi in x16:
        n += 1
        delta = f16(xi - mean)
        mean = f16(mean + f16(delta / f16(n)))
        delta2 = f16(xi - mean)
        M2 = f16(M2 + f16(delta * delta2))
    return mean, f16(M2 / f16(n))


def _two_pass_mean_var(x16):
    n = len(x16)
    s = f16(0.0)
    for xi in x16:
        s = f16(s + xi)
    mean = f16(s / f16(n))
    s2 = f16(0.0)
    for xi in x16:
        d = f16(xi - mean)
        s2 = f16(s2 + f16(d * d))
    return mean, f16(s2 / f16(n))


def _normalize(x16, mean, var, g16, b16, eps):
    denom = f16(np.sqrt(f16(var + f16(eps))))
    inv = f16(f16(1.0) / denom)
    out = np.empty(len(x16), dtype=f16)
    for i in range(len(x16)):
        xhat = f16(f16(x16[i] - mean) * inv)
        out[i] = f16(f16(g16[i] * xhat) + b16[i])
    return out


def layernorm_fp16_welford(x, gamma, beta, eps=1e-5):
    x16 = np.asarray(x, dtype=f16)
    g16 = np.asarray(gamma, dtype=f16)
    b16 = np.asarray(beta, dtype=f16)
    mean, var = _welford_mean_var(x16)
    return _normalize(x16, mean, var, g16, b16, eps)


def layernorm_fp16_two_pass(x, gamma, beta, eps=1e-5):
    x16 = np.asarray(x, dtype=f16)
    g16 = np.asarray(gamma, dtype=f16)
    b16 = np.asarray(beta, dtype=f16)
    mean, var = _two_pass_mean_var(x16)
    return _normalize(x16, mean, var, g16, b16, eps)
