import numpy as np


def autoround_block(W: np.ndarray, bits: int, steps: int, lr: float, seed: int):
    np.random.seed(seed)
    W = np.asarray(W, dtype=np.float64)

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = np.max(np.abs(W)) / qmax

    r = np.zeros_like(W, dtype=np.float64)

    def quantize(offset):
        q = np.clip(np.floor(W / scale + offset + 0.5), qmin, qmax)
        return q * scale

    best = quantize(r)
    best_mse = float(np.mean((best - W) ** 2))

    for _ in range(steps):
        current = quantize(r)
        current_mse = float(np.mean((current - W) ** 2))
        if current_mse < best_mse:
            best = current
            best_mse = current_mse

        grad = (2.0 * scale / W.size) * (current - W)
        r = r - lr * np.sign(grad)

    current = quantize(r)
    current_mse = float(np.mean((current - W) ** 2))
    if current_mse < best_mse:
        best = current
        best_mse = current_mse

    return best.astype(np.float64), best_mse
