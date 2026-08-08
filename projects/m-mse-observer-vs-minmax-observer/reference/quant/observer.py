import numpy as np


def minmax_observe(tensor, qmin, qmax, symmetric=False):
    x = np.asarray(tensor, dtype=np.float64)
    if symmetric:
        max_abs = float(np.max(np.abs(x)))
        if max_abs == 0.0:
            scale = 1.0
            zero_point = 0
        else:
            bound = max(abs(qmin), abs(qmax))
            scale = max_abs / float(bound)
            zero_point = 0
    else:
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        if max_val == min_val:
            scale = 1.0
            zero_point = int(qmin)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            zp_real = float(qmin) - min_val / scale
            zero_point = int(np.clip(np.round(zp_real), qmin, qmax))
    return {"scale": float(scale), "zero_point": int(zero_point)}


def mse_observe(tensor, qmin, qmax, symmetric=False, grid_steps=100):
    x = np.asarray(tensor, dtype=np.float64)
    if symmetric:
        max_abs = float(np.max(np.abs(x)))
        if max_abs == 0.0:
            return {"scale": 1.0, "zero_point": 0}
        bound = max(abs(qmin), abs(qmax))
        best_scale = max_abs / float(bound)
        best_mse = float("inf")
        best_zp = 0

        candidates = np.linspace(0.1 * max_abs, max_abs, grid_steps)
        for cand in candidates:
            scale = cand / float(bound)
            if scale <= 0:
                continue
            q = np.clip(np.round(x / scale), qmin, qmax)
            deq = q * scale
            mse = float(np.mean((x - deq) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
        return {"scale": float(best_scale), "zero_point": 0}
    else:
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        if max_val == min_val:
            return {"scale": 1.0, "zero_point": int(qmin)}

        full_range = max_val - min_val
        best_scale = full_range / float(qmax - qmin)
        best_zp = int(np.clip(np.round(float(qmin) - min_val / best_scale), qmin, qmax))
        best_mse = float("inf")

        fractions = np.linspace(0.1, 1.0, grid_steps)
        for frac in fractions:
            curr_range = full_range * frac
            if curr_range <= 0:
                continue
            scale = curr_range / float(qmax - qmin)
            center = (min_val + max_val) / 2.0
            c_min = center - curr_range / 2.0
            zp_real = float(qmin) - c_min / scale
            zp = int(np.clip(np.round(zp_real), qmin, qmax))

            q = np.clip(np.round(x / scale + zp), qmin, qmax)
            deq = (q - zp) * scale
            mse = float(np.mean((x - deq) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
                best_zp = zp
        return {"scale": float(best_scale), "zero_point": int(best_zp)}
