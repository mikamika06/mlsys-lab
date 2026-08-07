import numpy as np

def get_fixture():
    np.random.seed(42)
    in_dim = 128
    out_dim = 64
    batch = 256

    w = np.random.randn(out_dim, in_dim)
    x = np.random.randn(in_dim, batch)

    scales = np.exp(np.random.randn(in_dim))
    x = x * scales[:, None]

    w_scales = 1.0 / np.sqrt(scales)
    w = w * w_scales[None, :]

    x_mean = np.random.randn(in_dim) * 2
    x = x + x_mean[:, None]

    return w, x

def oracle_compare(w, x, sparsity):
    k = max(1, int(w.shape[1] * (1.0 - sparsity)))

    s_mag = np.abs(w)
    w_mag = np.zeros_like(w)
    for i in range(w.shape[0]):
        idx = np.argsort(s_mag[i])[-k:]
        w_mag[i, idx] = w[i, idx]
    mse_mag = np.mean((w @ x - w_mag @ x)**2)

    x_norm = np.linalg.norm(x, axis=1)
    s_wan = np.abs(w) * x_norm[None, :]
    w_wan = np.zeros_like(w)
    for i in range(w.shape[0]):
        idx = np.argsort(s_wan[i])[-k:]
        w_wan[i, idx] = w[i, idx]

    x_mean = np.mean(x, axis=1)
    bias_wan = (w - w_wan) @ x_mean
    y_pred = w_wan @ x + bias_wan[:, None]
    mse_wan = np.mean((w @ x - y_pred)**2)

    return float(mse_mag), float(mse_wan)
