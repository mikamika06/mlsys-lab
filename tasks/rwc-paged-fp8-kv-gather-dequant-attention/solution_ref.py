import math
import numpy as np


def _decode_e4m3(code: np.ndarray) -> np.ndarray:
    """Real E4M3FN bit-pattern decode: 1 sign, 4 exponent, 3 mantissa bits."""
    code = np.asarray(code, dtype=np.uint8)
    shape = code.shape
    flat_code = code.ravel()
    out_list = []
    
    for c in flat_code:
        sign = -1.0 if (c & 0x80) != 0 else 1.0
        e = int((c >> 3) & 0x0F)
        m = int(c & 0x07)
        
        if e == 15 and m == 7:
            out_list.append(float('nan'))
        elif e == 0:
            subnormal = sign * (m / 8.0) * math.exp2(-6.0)
            out_list.append(subnormal)
        else:
            normal = sign * (1.0 + m / 8.0) * math.exp2(float(e - 7))
            out_list.append(normal)
            
    return np.array(out_list, dtype=np.float64).reshape(shape)


def paged_fp8_attention(k_codes_phys, v_codes_phys, k_scale, v_scale, block_table, seq_len, q):
    """Gather paged FP8 (E4M3FN) KV via block_table, truncate to seq_len,
    dequantize per head, and run scaled dot-product attention."""
    _, block_size, n_heads, D = k_codes_phys.shape
    L_b = block_table.shape[0]

    k_codes = k_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]
    v_codes = v_codes_phys[block_table].reshape(L_b * block_size, n_heads, D)[:seq_len]

    k_scale = np.asarray(k_scale, dtype=np.float64)
    v_scale = np.asarray(v_scale, dtype=np.float64)
    k_deq = _decode_e4m3(k_codes) * k_scale[None, :, None]
    v_deq = _decode_e4m3(v_codes) * v_scale[None, :, None]

    q = np.asarray(q, dtype=np.float64)
    scale = 1.0 / math.sqrt(D)

    out = np.zeros((n_heads, D), dtype=np.float64)
    for h in range(n_heads):
        K = k_deq[:, h, :]
        V = v_deq[:, h, :]
        qh = q[h]
        
        seq_len_curr = K.shape[0]
        s = np.zeros(seq_len_curr, dtype=np.float64)
        for i in range(seq_len_curr):
            dot_val = 0.0
            for d in range(D):
                dot_val += K[i, d] * qh[d]
            s[i] = dot_val * scale
            
        max_s = s[0]
        for i in range(1, seq_len_curr):
            if s[i] > max_s:
                max_s = s[i]
                
        w = np.zeros(seq_len_curr, dtype=np.float64)
        sum_w = 0.0
        for i in range(seq_len_curr):
            val = math.exp(s[i] - max_s)
            w[i] = val
            sum_w += val
            
        for i in range(seq_len_curr):
            w[i] /= sum_w
            
        for d in range(D):
            acc = 0.0
            for i in range(seq_len_curr):
                acc += w[i] * V[i, d]
            out[h, d] = acc

    return out
