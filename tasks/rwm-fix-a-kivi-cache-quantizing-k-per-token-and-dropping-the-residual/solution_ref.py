import numpy as np


def _quantize_dequant(x, scale):
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q.astype(np.float64) * scale


def quantize_dequant_kv_cache(K: np.ndarray, V: np.ndarray, R: int):
    t_cut = K.shape[2] - R

    K_out = np.empty_like(K, dtype=np.float64)
    V_out = np.empty_like(V, dtype=np.float64)

    K_main = K[:, :, :t_cut, :]
    k_scale = np.max(np.abs(K_main), axis=2, keepdims=True) / 127.0
    K_out[:, :, :t_cut, :] = _quantize_dequant(K_main, k_scale)

    V_main = V[:, :, :t_cut, :]
    v_scale = np.max(np.abs(V_main), axis=3, keepdims=True) / 127.0
    V_out[:, :, :t_cut, :] = _quantize_dequant(V_main, v_scale)

    K_out[:, :, t_cut:, :] = K[:, :, t_cut:, :].astype(np.float16).astype(np.float64)
    V_out[:, :, t_cut:, :] = V[:, :, t_cut:, :].astype(np.float16).astype(np.float64)

    return K_out, V_out
