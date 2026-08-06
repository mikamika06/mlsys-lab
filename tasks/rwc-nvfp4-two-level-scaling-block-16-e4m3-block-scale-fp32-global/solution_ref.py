import numpy as np


def _e4m3_values():
    vals = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exp = (bits >> 3) & 0x0F
        mant = bits & 0x07
        if exp == 0:
            vals.append(sign * (mant / 8.0) * 2 ** -6)
        elif exp < 15:
            vals.append(sign * (1.0 + mant / 8.0) * 2 ** (exp - 7))
        else:
            vals.append(sign * 448.0)
    return np.array(vals, dtype=np.float64)


_E4 = _e4m3_values()
_E2 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0],
    dtype=np.float64,
)


def _encode_e4m3(x):
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty(x_arr.shape, dtype=np.uint8)
    for idx_flat in range(x_arr.size):
        val = x_arr.flat[idx_flat]
        best_i = 0
        min_diff = abs(val - _E4[0])
        for i in range(1, len(_E4)):
            diff = abs(val - _E4[i])
            if diff < min_diff:
                min_diff = diff
                best_i = i
        out.flat[idx_flat] = best_i
    return out


def _decode_e4m3(c):
    c_arr = np.asarray(c, dtype=np.uint8)
    out = np.empty(c_arr.shape, dtype=np.float64)
    for idx_flat in range(c_arr.size):
        out.flat[idx_flat] = _E4[c_arr.flat[idx_flat]]
    return out


def quantize_nvfp4(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    
    blocks_list = []
    for i in range(0, n, 16):
        chunk = x[i:i + 16]
        m = 0.0
        for val in chunk:
            abs_val = float(val) if val >= 0 else -float(val)
            if abs_val > m:
                m = abs_val
        blocks_list.append(m)
    blocks = np.asarray(blocks_list, dtype=np.float64)

    max_block = 0.0
    for b_val in blocks:
        if b_val > max_block:
            max_block = b_val

    global_scale = float(max_block / 448.0) if max_block != 0 else 1.0
    
    raw = np.empty(blocks.shape, dtype=np.float64)
    for idx_flat in range(blocks.size):
        raw.flat[idx_flat] = blocks.flat[idx_flat] / global_scale
    
    block_scales = _encode_e4m3(raw)
    decoded = _decode_e4m3(block_scales)

    codes = np.empty(n, dtype=np.uint8)
    reconstruction = np.empty(n, dtype=np.float32)
    
    for b in range(len(decoded)):
        start = b * 16
        end = min(n, start + 16)
        
        dec_val = decoded[b]
        denom = dec_val * global_scale
        
        for j in range(start, end):
            q_val = float(x[j]) / denom
            
            best_i = 0
            min_diff = abs(q_val - _E2[0])
            for i in range(1, len(_E2)):
                diff = abs(q_val - _E2[i])
                if diff < min_diff:
                    min_diff = diff
                    best_i = i
            
            codes[j] = best_i
            reconstruction[j] = np.float32(_E2[best_i] * denom)
            
    return codes, block_scales, global_scale, reconstruction
