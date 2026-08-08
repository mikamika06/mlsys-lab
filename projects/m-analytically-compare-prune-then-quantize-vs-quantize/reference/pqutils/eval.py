import numpy as np


def measure_reconstruction_error(original, compressed):
    orig = np.array(original, dtype=np.float64)
    comp = np.array(compressed, dtype=np.float64)
    mse = np.mean((orig - comp)**2)
    mae = np.mean(np.abs(orig - comp))
    max_err = np.max(np.abs(orig - comp))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max": float(max_err)
    }
