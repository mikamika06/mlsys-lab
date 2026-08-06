import numpy as np
import math

E4M3_MAX = 448.0


def _e4m3_grid() -> np.ndarray:
    bias = 7
    vals = set()
    for sign in (1.0, -1.0):
        for e in range(16):
            for m in range(8):
                if e == 15 and m == 7:
                    continue  # NaN
                if e == 0:
                    v = (m / 8.0) * (2.0 ** (1 - bias))
                else:
                    v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
                vals.add(sign * v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    flat = x.reshape(-1)
    
    rounded_list = []
    for val in flat:
        best_diff = float("inf")
        best_grid_val = _GRID[0]
        for g in _GRID:
            diff = g - val
            if diff < 0.0:
                diff = -diff
            if diff < best_diff:
                best_diff = diff
                best_grid_val = g
            elif diff == best_diff:
                if g > best_grid_val:
                    best_grid_val = g
        rounded_list.append(best_grid_val)
        
    return np.array(rounded_list, dtype=np.float64).reshape(original_shape)


def qfloat8_weight_quant(W: np.ndarray):
    """Per-tensor scale (to E4M3's max magnitude 448) + nearest-E4M3 cast.

    Returns (scale, e4m3_values, W_hat).
    """
    W = np.asarray(W, dtype=np.float64)
    flat_W = W.reshape(-1)
    
    amax = 0.0
    for val in flat_W:
        abs_val = val if val >= 0.0 else -val
        if abs_val > amax:
            amax = abs_val
            
    scale = amax / E4M3_MAX if amax > 0 else 1.0
    
    W_scaled = np.empty_like(W)
    flat_W_scaled = W_scaled.reshape(-1)
    for i in range(flat_W.shape[0]):
        flat_W_scaled[i] = flat_W[i] / scale
        
    codes = _round_to_e4m3(W_scaled)
    
    W_hat = np.empty_like(W)
    flat_codes = codes.reshape(-1)
    flat_W_hat = W_hat.reshape(-1)
    for i in range(flat_codes.shape[0]):
        flat_W_hat[i] = flat_codes[i] * scale
        
    return scale, codes, W_hat
