import numpy as np


def _oracle_quantize_kv(kv, kv_bits, kv_group_size):
    x = np.asarray(kv, dtype=np.float64)
    groups = x.shape[-1] // kv_group_size
    new_shape = x.shape[:-1] + (groups, kv_group_size)
    g = x.reshape(new_shape)
    qmax = (1 << kv_bits) - 1
    xmin = np.min(g, axis=-1, keepdims=True)
    xmax = np.max(g, axis=-1, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.round(-xmin / scale)
    zero = np.clip(zero, 0, qmax)
    q = np.round(g / scale + zero)
    q = np.clip(q, 0, qmax).astype(np.int32)
    return q, scale, zero


def _oracle_dequant(q, scale, zero, kv_group_size):
    x = scale * (q.astype(np.float64) - zero)
    return x.reshape(x.shape[:-2] + (x.shape[-2] * kv_group_size,))


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[[0.1, -0.3, 1.5, 2.2]]]), 4, 2),
        (np.array([[[1.0, 2.0, 8.0, 10.0, -3.0, 0.0, 4.0, 5.0]]]), 3, 2),
        (np.arange(32, dtype=np.float64).reshape(2, 2, 8) / 7.0, 8, 4),
    ]

    max_mse = 0.0
    max_attn = 0.0

    for kv, bits, group_size in cases:
        try:
            q, scales, zeros = sol.quantize_kv_group_affine(
                kv.copy(), bits, group_size
            )
            q = np.asarray(q)
            scales = np.asarray(scales)
            zeros = np.asarray(zeros)
        except Exception:
            return {"mse": 1.0, "attention_mse": 1.0}

        ref_q, ref_s, ref_z = _oracle_quantize_kv(kv, bits, group_size)

        got = _oracle_dequant(q, scales, zeros, group_size)
        ref = _oracle_dequant(ref_q, ref_s, ref_z, group_size)

        mse = float(np.mean((got - ref) ** 2))
        max_mse = max(max_mse, mse)

        query = np.arange(kv.shape[-1], dtype=np.float64).reshape(1, -1)
        attn_got = query @ got.reshape(-1, kv.shape[-1]).T
        attn_ref = query @ ref.reshape(-1, kv.shape[-1]).T
        attn_mse = float(np.mean((attn_got - attn_ref) ** 2))
        max_attn = max(max_attn, attn_mse)

    return {"mse": max_mse, "attention_mse": max_attn}
