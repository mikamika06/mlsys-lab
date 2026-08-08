import numpy as np

def get_view(w, granularity, group_size=None):
    if granularity == "tensor":
        return w.reshape(1, -1)
    elif granularity == "axis_0":
        return w.reshape(w.shape[0], -1)
    elif granularity == "axis_1":
        return w.T.reshape(w.shape[1], -1)
    elif granularity == "group":
        return w.reshape(w.shape[0] * (w.shape[1] // group_size), group_size)
    raise ValueError("Invalid granularity")

def restore_view(q_view, w_shape, granularity):
    if granularity == "tensor":
        return q_view.reshape(w_shape)
    elif granularity == "axis_0":
        return q_view.reshape(w_shape)
    elif granularity == "axis_1":
        return q_view.reshape(w_shape[1], w_shape[0]).T
    elif granularity == "group":
        return q_view.reshape(w_shape)
    raise ValueError("Invalid granularity")

def calc_qparams(w_view, symmetric):
    if symmetric:
        max_abs = np.max(np.abs(w_view), axis=1, keepdims=True)
        scale = max_abs / 127.0
        scale = np.maximum(scale, 1e-9)
        zp = np.zeros_like(scale)
    else:
        w_min = np.minimum(np.min(w_view, axis=1, keepdims=True), 0.0)
        w_max = np.maximum(np.max(w_view, axis=1, keepdims=True), 0.0)
        scale = (w_max - w_min) / 255.0
        scale = np.maximum(scale, 1e-9)
        zp = np.round(-w_min / scale)
    return scale, zp

def apply_quant(w_view, scale, zp, symmetric):
    q = np.round(w_view / scale) + zp
    if symmetric:
        return np.clip(q, -127, 127).astype(np.int8)
    else:
        return np.clip(q, 0, 255).astype(np.uint8)

def apply_dequant(q_view, scale, zp):
    return (q_view.astype(np.float32) - zp) * scale

def evaluate_ladder(w, group_size=32):
    results = []
    for gran in ["tensor", "axis_0", "axis_1", "group"]:
        w_view = get_view(w, gran, group_size)
        scale, zp = calc_qparams(w_view, symmetric=False)
        q_view = apply_quant(w_view, scale, zp, symmetric=False)
        w_approx_view = apply_dequant(q_view, scale, zp)
        w_approx = restore_view(w_approx_view, w.shape, gran)

        meta_bytes = scale.size * 2 + zp.size * 1
        max_abs_err = float(np.max(np.abs(w - w_approx)))

        results.append({
            "granularity": gran,
            "meta_bytes": meta_bytes,
            "max_abs_err": max_abs_err
        })
    return results

def generate_fixture(seed=42, shape=(64, 128)):
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape).astype(np.float32) * 2.0
    # Add severe outliers to differentiate granularities
    w[10, :] *= 5.0
    w[:, 20] *= 5.0
    return w

FIXTURE_W = generate_fixture()
