import numpy as np


def _create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        exp_val = 10.0 ** (-(max_exponent_bits - 1) + i)
        data.extend((exp_val * means).tolist())
        if signed:
            data.extend((-exp_val * means).tolist())
    data.append(0.0)
    data.append(1.0)
    data = sorted(data)
    arr = np.array(data, dtype=np.float64)
    assert arr.shape[0] == 2 ** total_bits
    return arr


_MAP = _create_dynamic_map()


def quantize_dequantize_v_blockwise(v: np.ndarray, blocksize: int):
    """
    Build the 256-entry dynamic exponent map, quantize `v` blockwise
    (per-block absmax normalize -> nearest map code), dequantize, and
    return (v_hat, codes, absmax).
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    
    if n > 0:
        n_blocks = (n + blocksize - 1) // blocksize
    else:
        n_blocks = 0

    codes_list = []
    absmax_list = []
    v_hat_list = []

    for b in range(n_blocks):
        lo = b * blocksize
        hi = min((b + 1) * blocksize, n)
        
        amax = 0.0
        for i in range(lo, hi):
            val = v[i]
            if val < 0.0:
                val = -val
            if val > amax:
                amax = val

        if amax == 0.0:
            amax = 1.0

        absmax_list.append(amax)

        for i in range(lo, hi):
            normed = v[i] / amax
            
            best_idx = 0
            best_diff = -1.0
            
            for j in range(256):
                map_val = _MAP[j]
                diff = normed - map_val
                if diff < 0.0:
                    diff = -diff
                
                if j == 0 or diff < best_diff:
                    best_diff = diff
                    best_idx = j
                elif diff == best_diff:
                    pass

            codes_list.append(best_idx)
            v_hat_list.append(_MAP[best_idx] * amax)

    codes = np.array(codes_list, dtype=np.uint8)
    absmax = np.array(absmax_list, dtype=np.float32)
    v_hat = np.array(v_hat_list, dtype=np.float64)

    return v_hat, codes, absmax
