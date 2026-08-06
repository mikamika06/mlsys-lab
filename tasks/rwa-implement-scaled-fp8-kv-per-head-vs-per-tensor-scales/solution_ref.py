import math
import numpy as np


def _e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    shape = x.shape
    for idx in np.ndindex(shape):
        val = x[idx]
        ax = abs(val)
        if ax > 0:
            vals = min(ax, 448.0)
            exp = max(math.floor(math.log2(vals)), -6)
            base = math.pow(2.0, exp)
            mant = vals / base - 1.0
            mant_q = round(mant * 8.0) / 8.0
            vals_q = min(base * (1.0 + mant_q), 448.0)
            sign = 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
            out[idx] = sign * vals_q
    return out


def _qd(x, per_head):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    if per_head:
        scale = np.zeros((shape[0], 1, 1), dtype=np.float64)
        for i in range(shape[0]):
            max_val = 0.0
            for j in range(shape[1]):
                for k in range(shape[2]):
                    val = abs(x[i, j, k])
                    if val > max_val:
                        max_val = val
            s = max_val / 448.0
            if s < 1e-12:
                s = 1e-12
            scale[i, 0, 0] = s
    else:
        max_val = 0.0
        for idx in np.ndindex(shape):
            val = abs(x[idx])
            if val > max_val:
                max_val = val
        scale = max_val / 448.0
        if scale < 1e-12:
            scale = 1e-12

    x_scaled = np.zeros_like(x)
    for idx in np.ndindex(shape):
        if per_head:
            i = idx[0]
            x_scaled[idx] = x[idx] / scale[i, 0, 0]
        else:
            x_scaled[idx] = x[idx] / scale

    quantized = _e4m3(x_scaled)

    out = np.zeros_like(x)
    for idx in np.ndindex(shape):
        if per_head:
            i = idx[0]
            out[idx] = quantized[idx] * scale[i, 0, 0]
        else:
            out[idx] = quantized[idx] * scale
    return out


def scaled_fp8_kv_attention(K, V, Q, per_head):
    Kd = _qd(K, per_head)
    Vd = _qd(V, per_head)
    
    Q_f = np.asarray(Q, dtype=np.float64)
    b_sz, q_seq, q_dim = Q_f.shape
    _, k_seq, k_dim = Kd.shape
    _, _, v_dim = Vd.shape
    
    logits = np.zeros((b_sz, q_seq, k_seq), dtype=np.float64)
    for b in range(b_sz):
        for i in range(q_seq):
            for j in range(k_seq):
                acc = 0.0
                for d in range(q_dim):
                    acc += Q_f[b, i, d] * Kd[b, j, d]
                logits[b, i, j] = acc

    scale_factor = math.sqrt(K.shape[-1])
    for idx in np.ndindex(logits.shape):
        logits[idx] = logits[idx] / scale_factor

    for b in range(b_sz):
        for i in range(q_seq):
            row_max = logits[b, i, 0]
            for j in range(1, k_seq):
                if logits[b, i, j] > row_max:
                    row_max = logits[b, i, j]
            for j in range(k_seq):
                logits[b, i, j] = logits[b, i, j] - row_max

    probs = np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        probs[idx] = math.exp(logits[idx])

    for b in range(b_sz):
        for i in range(q_seq):
            row_sum = 0.0
            for j in range(k_seq):
                row_sum += probs[b, i, j]
            for j in range(k_seq):
                probs[b, i, j] = probs[b, i, j] / row_sum

    out = np.zeros((b_sz, q_seq, v_dim), dtype=np.float64)
    for b in range(b_sz):
        for i in range(q_seq):
            for d in range(v_dim):
                acc = 0.0
                for k in range(k_seq):
                    acc += probs[b, i, k] * Vd[b, k, d]
                out[b, i, d] = acc

    return out
