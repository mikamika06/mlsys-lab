import numpy as np
from quant_observe.observer import minmax_observer, mse_observer

def ignored_zp_bias(x: np.ndarray, args: dict, method: str) -> float:
    if method == "mse":
        scale, zp = mse_observer(x, args)
    else:
        scale, zp = minmax_observer(x, args)

    bits = args["bits"]
    sym = args["symmetric"]
    qmin = -(2**(bits-1)) if sym else 0
    qmax = 2**(bits-1) - 1 if sym else 2**bits - 1

    xq = np.clip(np.round(x / scale) + zp, qmin, qmax)
    x_correct = (xq - zp) * scale
    x_ignored = xq * scale
    return float(np.sum(x_ignored) - np.sum(x_correct))
