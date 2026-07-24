import numpy as np


def quantize_dequant_kv_cache(K: np.ndarray, V: np.ndarray, R: int):
    # Broken: uses per-token K scales and drops the fp16 residual window.
    t_cut = K.shape[2] - R

    K_main = K[:, :, :t_cut, :]
    K_scale = np.max(np.abs(K_main), axis=3, keepdims=True) / 127.0
    K_out = np.empty_like(K, dtype=np.float64)
    K_out[:, :, :t_cut, :] = (
        np.clip(np.round(K_main / np.where(K_scale == 0, 1.0, K_scale)), -127, 127)
        .astype(np.int8)
        .astype(np.float64)
        * np.where(K_scale == 0, 1.0, K_scale)
    )

    K_out[:, :, t_cut:, :] = K[:, :, t_cut:, :]

    V_scale = np.max(np.abs(V), axis=3, keepdims=True) / 127.0
    V_out = (
        np.clip(np.round(V / np.where(V_scale == 0, 1.0, V_scale)), -127, 127)
        .astype(np.int8)
        .astype(np.float64)
        * np.where(V_scale == 0, 1.0, V_scale)
    )

    return K_out, V_out
