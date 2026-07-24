import numpy as np
from mlsys import scorers


def _quantize(x, block_size):
    x = np.asarray(x, dtype=np.float32)
    codes = np.zeros(x.shape, dtype=np.int8)
    scales = []
    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]
        scale = float(np.max(np.abs(block)) / 127.0) if len(block) else 1.0
        if scale == 0:
            scale = 1.0
        q = np.round(block / scale).clip(-127, 127).astype(np.int8)
        codes[start:start + len(block)] = q
        scales.append(scale)
    return codes, np.asarray(scales, dtype=np.float32)


def _dequantize(codes, scales, block_size):
    out = np.zeros(len(codes), dtype=np.float32)
    for i, scale in enumerate(scales):
        start = i * block_size
        end = min(len(codes), start + block_size)
        out[start:end] = codes[start:end].astype(np.float32) * scale
    return out


def _ref_step(params, grads, state, lr, beta1, beta2, eps, wd, block_size):
    if state is None:
        m = np.zeros_like(params, dtype=np.float32)
        v = np.zeros_like(params, dtype=np.float32)
        step = 0
    else:
        m = _dequantize(state["m_codes"], state["m_scales"], block_size)
        v = _dequantize(state["v_codes"], state["v_scales"], block_size)
        step = int(state["step"])

    step += 1
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads * grads)

    mh = m / (1 - beta1 ** step)
    vh = v / (1 - beta2 ** step)
    params = params - lr * mh / (np.sqrt(vh) + eps) - lr * wd * params

    mc, ms = _quantize(m, block_size)
    vc, vs = _quantize(v, block_size)
    return params.astype(np.float32), {
        "m_codes": mc,
        "m_scales": ms,
        "v_codes": vc,
        "v_scales": vs,
        "step": step,
    }


def grade(sol, fx) -> dict:
    params0 = np.array([1.0, -2.0, 0.5, 4.0, -3.0, 2.0], dtype=np.float32)
    grads = [
        np.array([0.2, -0.1, 0.05, 0.4, -0.3, 0.1], dtype=np.float32),
        np.array([-0.1, 0.3, 0.2, -0.2, 0.05, -0.4], dtype=np.float32),
        np.array([0.15, -0.25, 0.1, 0.05, 0.2, -0.1], dtype=np.float32),
        np.array([-0.05, 0.1, -0.2, 0.3, -0.15, 0.25], dtype=np.float32),
    ]
    args = (0.01, 0.9, 0.999, 1e-8, 0.01, 2)

    ref_p = params0.copy()
    ref_s = None
    got_p = params0.copy()
    got_s = None

    try:
        for g in grads:
            ref_p, ref_s = _ref_step(ref_p, g, ref_s, *args)
            got_p, got_s = sol.adamw_8bit_step(got_p, g, got_s, *args)
        err = scorers.rel_err(ref_p, got_p)
    except Exception:
        err = float("inf")

    return {"rel_err": float(err)}
