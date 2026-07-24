import numpy as np


def _quantize_dequant(x, scale, axis):
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q.astype(np.float64) * scale


def _oracle(K, V, R):
    t_cut = K.shape[2] - R

    K_out = np.empty_like(K, dtype=np.float64)
    V_out = np.empty_like(V, dtype=np.float64)

    K_main = K[:, :, :t_cut, :]
    K_scale = np.max(np.abs(K_main), axis=2, keepdims=True) / 127.0
    K_out[:, :, :t_cut, :] = _quantize_dequant(K_main, K_scale, axis=2)

    V_main = V[:, :, :t_cut, :]
    V_scale = np.max(np.abs(V_main), axis=3, keepdims=True) / 127.0
    V_out[:, :, :t_cut, :] = _quantize_dequant(V_main, V_scale, axis=3)

    K_out[:, :, t_cut:, :] = K[:, :, t_cut:, :].astype(np.float16).astype(np.float64)
    V_out[:, :, t_cut:, :] = V[:, :, t_cut:, :].astype(np.float16).astype(np.float64)

    return K_out, V_out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    K = rng.normal(size=(2, 3, 9, 5)).astype(np.float32)
    V = rng.normal(size=(2, 3, 9, 5)).astype(np.float32)
    R = 3

    ref_k, ref_v = _oracle(K, V, R)

    try:
        got_k, got_v = sol.quantize_dequant_kv_cache(K, V, R)
        got = np.concatenate(
            [np.asarray(got_k).ravel(), np.asarray(got_v).ravel()]
        )
        ref = np.concatenate([ref_k.ravel(), ref_v.ravel()])
        err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))

        residual_ref = np.concatenate(
            [
                ref_k[:, :, -R:, :].ravel(),
                ref_v[:, :, -R:, :].ravel(),
            ]
        )
        residual_got = np.concatenate(
            [
                np.asarray(got_k)[:, :, -R:, :].ravel(),
                np.asarray(got_v)[:, :, -R:, :].ravel(),
            ]
        )
        residual_exact = float(np.array_equal(residual_got, residual_ref))
    except Exception:
        return {"rel_err": float("inf"), "residual_exact": 0.0}

    return {"rel_err": err, "residual_exact": residual_exact}
