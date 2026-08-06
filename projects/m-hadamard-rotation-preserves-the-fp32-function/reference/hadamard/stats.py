import numpy as np

def get_outlier_stats(x, h):
    x_rot = np.matmul(x, h)
    return {
        "orig_max": float(np.max(np.abs(x))),
        "rot_max": float(np.max(np.abs(x_rot))),
        "orig_kurtosis": float(np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-8)),
        "rot_kurtosis": float(np.mean((x_rot - np.mean(x_rot))**4) / (np.std(x_rot)**4 + 1e-8))
    }
