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
    n_blocks = int(np.ceil(n / blocksize)) if n > 0 else 0
    codes = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    v_hat = np.zeros(n, dtype=np.float64)

    for b in range(n_blocks):
        lo, hi = b * blocksize, min((b + 1) * blocksize, n)
        block = v[lo:hi]
        amax = float(np.max(np.abs(block))) if block.size > 0 else 0.0
        amax = amax if amax > 0 else 1.0
        absmax[b] = amax
        normed = block / amax
        diffs = np.abs(normed[:, None] - _MAP[None, :])
        idx = np.argmin(diffs, axis=1).astype(np.uint8)
        codes[lo:hi] = idx
        v_hat[lo:hi] = _MAP[idx] * amax

    return v_hat, codes, absmax
