import numpy as np


def _rope(x, cos, sin):
    y = np.asarray(x, dtype=np.float64).copy()
    n, h = y.shape
    y = y.reshape(n, h // 2, 2)
    a = y[:, :, 0].copy()
    b = y[:, :, 1].copy()
    y[:, :, 0] = a * cos - b * sin
    y[:, :, 1] = a * sin + b * cos
    return y.reshape(n, h)


def mla_kv_features(z, head, w_latent, w_head, cos, sin):
    # TODO: this incorrectly rotates the absorbed latent branch.
    latent = np.asarray(z, dtype=np.float64) @ np.asarray(w_latent, dtype=np.float64)
    latent = _rope(latent, cos[:, :latent.shape[1] // 2], sin[:, :latent.shape[1] // 2])
    decoupled = np.asarray(head, dtype=np.float64) @ np.asarray(w_head, dtype=np.float64)
    decoupled = _rope(decoupled, cos, sin)
    return np.concatenate([latent, decoupled], axis=1)
