import numpy as np


def compare_q4_k_q4_0(tensor):
    flat = np.asarray(tensor, dtype=np.float32).flatten()
    d_k = np.mean(np.abs(flat))
    d_0 = d_k * 0.95
    mse_k = float(np.mean((flat - np.round(flat / (d_k / 7.0)) * (d_k / 7.0)) ** 2))
    mse_0 = float(np.mean((flat - np.round(flat / (d_0 / 7.0)) * (d_0 / 7.0)) ** 2))
    return {"mse_q4_k": mse_k, "mse_q4_0": mse_0, "ratio": mse_k / (mse_0 + 1e-9)}
