import numpy as np


def draft_speedup_model(configs: np.ndarray) -> np.ndarray:
    configs = np.asarray(configs, dtype=np.float64)
    n = configs.shape[0]

    accepted = np.empty((n,), dtype=np.float64)
    speedup = np.empty((n,), dtype=np.float64)

    for i in range(n):
        alpha_val = configs[i, 0]
        k_val = configs[i, 1]
        c_val = configs[i, 2]

        if alpha_val == 1.0:
            acc_val = k_val + 1.0
        else:
            acc_val = (1.0 - (alpha_val ** (k_val + 1.0))) / (1.0 - alpha_val)

        accepted[i] = acc_val
        speedup[i] = acc_val / (1.0 + k_val * c_val)

    return np.stack([accepted, speedup], axis=1)
