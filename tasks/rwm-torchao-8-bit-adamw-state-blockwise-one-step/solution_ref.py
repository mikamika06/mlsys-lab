import numpy as np


def _quantize(x, block_size):
    codes = np.zeros(len(x), dtype=np.int8)
    scales = []
    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]
        scale = float(np.max(np.abs(block)) / 127.0)
        if scale == 0:
            scale = 1.0
        codes[start:start + len(block)] = np.round(
            block / scale
        ).clip(-127, 127).astype(np.int8)
        scales.append(scale)
    return codes, np.asarray(scales, dtype=np.float32)


def _dequantize(codes, scales, block_size):
    out = np.zeros(len(codes), dtype=np.float32)
    for i, scale in enumerate(scales):
        start = i * block_size
        end = min(len(codes), start + block_size)
        out[start:end] = codes[start:end].astype(np.float32) * scale
    return out


def adamw_8bit_step(params, grads, state, lr, beta1, beta2, eps, weight_decay, block_size):
    params = np.asarray(params, dtype=np.float32)
    grads = np.asarray(grads, dtype=np.float32)

    if state is None:
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        step = 0
    else:
        m = _dequantize(state["m_codes"], state["m_scales"], block_size)
        v = _dequantize(state["v_codes"], state["v_scales"], block_size)
        step = state["step"]

    step += 1
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads * grads

    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    params = params - lr * weight_decay * params

    mc, ms = _quantize(m, block_size)
    vc, vs = _quantize(v, block_size)

    return params, {
        "m_codes": mc,
        "m_scales": ms,
        "v_codes": vc,
        "v_scales": vs,
        "step": step,
    }
