import numpy as np
import math

def prune_then_quantize(W: np.ndarray, group_size: int) -> np.ndarray:
    """
    Prune a 2‑D weight matrix with a 2:4 mask and then quantize the surviving
    entries per block using signed int‑4 scaling. The result is dequantized.
    """
    W = np.asarray(W, dtype=np.float64)
    n, d = W.shape
    out = np.zeros_like(W)

    for i in range(n):
        row = W[i]
        for start in range(0, d, group_size):
            end = min(start + group_size, d)
            block = row[start:end].copy()
            blen = len(block)

            if blen <= 2:
                keep_idx = np.arange(blen)
            else:
                abs_vals = []
                for val in block:
                    if val < 0.0:
                        abs_vals.append(-val)
                    else:
                        abs_vals.append(val)
                
                max1_idx = 0
                max1_val = abs_vals[0]
                max2_idx = 1
                max2_val = abs_vals[1]
                if max2_val > max1_val:
                    max1_idx, max2_idx = max2_idx, max1_idx
                    max1_val, max2_val = max2_val, max1_val

                for idx in range(2, blen):
                    v = abs_vals[idx]
                    if v > max1_val:
                        max2_idx = max1_idx
                        max2_val = max1_val
                        max1_idx = idx
                        max1_val = v
                    elif v > max2_val:
                        max2_idx = idx
                        max2_val = v

                keep_idx = np.array([max2_idx, max1_idx], dtype=np.intp)
                if keep_idx[0] > keep_idx[1]:
                    keep_idx[0], keep_idx[1] = keep_idx[1], keep_idx[0]

            mask = np.zeros_like(block, dtype=bool)
            for idx in keep_idx:
                mask[idx] = True

            for idx in range(blen):
                if not mask[idx]:
                    block[idx] = 0.0

            max_abs = 0.0
            for val in block:
                av = -val if val < 0.0 else val
                if av > max_abs:
                    max_abs = av

            if max_abs == 0:
                for idx in range(blen):
                    out[i, start + idx] = 0.0
                continue

            scale = max_abs / 7.0
            for idx in range(blen):
                val = block[idx]
                div = val / scale
                q = math.floor(div + 0.5)
                if q < -8.0:
                    q = -8.0
                elif q > 7.0:
                    q = 7.0
                out[i, start + idx] = q * scale

    return out
