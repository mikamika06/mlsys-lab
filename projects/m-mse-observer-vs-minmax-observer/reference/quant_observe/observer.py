import numpy as np

def minmax_observer(x: np.ndarray, args: dict) -> tuple[float, float]:
    bits = args["bits"]
    sym = args["symmetric"]
    if sym:
        qmin = -(2**(bits-1))
        qmax = 2**(bits-1) - 1
        m = float(np.max(np.abs(x)))
        if m == 0:
            return 1.0, 0.0
        scale = m / qmax
        zp = 0.0
    else:
        qmin = 0
        qmax = 2**bits - 1
        m_min, m_max = float(np.min(x)), float(np.max(x))
        if m_min == m_max:
            return 1.0, 0.0
        scale = (m_max - m_min) / qmax
        zp = np.clip(np.round(-m_min / scale), qmin, qmax)
    return float(scale), float(zp)

def mse_observer(x: np.ndarray, args: dict) -> tuple[float, float]:
    b_scale, b_zp = minmax_observer(x, args)
    if b_scale == 1.0 and (np.max(x) == np.min(x)):
        return 1.0, 0.0

    alphas = np.linspace(0.1, 1.0, 100)
    best_mse = float('inf')
    best_scale = b_scale

    bits = args["bits"]
    sym = args["symmetric"]
    qmin = -(2**(bits-1)) if sym else 0
    qmax = 2**(bits-1) - 1 if sym else 2**bits - 1

    for a in alphas:
        s = b_scale * a
        xq = np.clip(np.round(x / s) + b_zp, qmin, qmax)
        x_approx = (xq - b_zp) * s
        mse = np.mean((x - x_approx)**2)
        if mse < best_mse:
            best_mse = mse
            best_scale = s

    return float(best_scale), float(b_zp)
