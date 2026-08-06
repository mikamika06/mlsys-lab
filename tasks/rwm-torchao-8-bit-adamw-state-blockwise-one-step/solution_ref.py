import math
import numpy as np


def _quantize(x, block_size):
    x = np.asarray(x, dtype=np.float32)
    codes = np.zeros(x.shape, dtype=np.int8)
    scales = []
    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]
        max_val = 0.0
        for val in block:
            abs_val = val if val >= 0 else -val
            if abs_val > max_val:
                max_val = abs_val
        scale = float(max_val / 127.0) if len(block) else 1.0
        if scale == 0:
            scale = 1.0
        for j in range(len(block)):
            val = block[j]
            q = round(val / scale)
            if q > 127:
                q = 127
            elif q < -127:
                q = -127
            codes[start + j] = int(q)
        scales.append(scale)
    return codes, np.asarray(scales, dtype=np.float32)


def _dequantize(codes, scales, block_size):
    out = np.zeros(len(codes), dtype=np.float32)
    for i, scale in enumerate(scales):
        start = i * block_size
        end = min(len(codes), start + block_size)
        for j in range(start, end):
            out[j] = float(codes[j]) * scale
    return out


def adamw_8bit_step(params, grads, state, lr, beta1, beta2, eps, weight_decay, block_size):
    params = np.asarray(params, dtype=np.float32)
    grads = np.asarray(grads, dtype=np.float32)

    if state is None:
        m = np.zeros(len(params), dtype=np.float32)
        v = np.zeros(len(params), dtype=np.float32)
        step = 0
    else:
        m = _dequantize(state["m_codes"], state["m_scales"], block_size)
        v = _dequantize(state["v_codes"], state["v_scales"], block_size)
        step = int(state["step"])

    step += 1
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step

    new_params = np.zeros(len(params), dtype=np.float32)
    m_new = np.zeros(len(params), dtype=np.float32)
    v_new = np.zeros(len(params), dtype=np.float32)

    for i in range(len(params)):
        p = params[i]
        g = grads[i]
        mi = m[i]
        vi = v[i]

        mi_next = beta1 * mi + (1.0 - beta1) * g
        vi_next = beta2 * vi + (1.0 - beta2) * g * g

        m_new[i] = mi_next
        v_new[i] = vi_next

        mh = mi_next / (1.0 - beta1_pow)
        vh = vi_next / (1.0 - beta2_pow)

        new_params[i] = p - lr * mh / (math.sqrt(vh) + eps) - lr * weight_decay * p

    mc, ms = _quantize(m_new, block_size)
    vc, vs = _quantize(v_new, block_size)

    return new_params, {
        "m_codes": mc,
        "m_scales": ms,
        "v_codes": vc,
        "v_scales": vs,
        "step": step,
    }
