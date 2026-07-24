import numpy as np


def draft_speedup_model(configs: np.ndarray) -> np.ndarray:
    configs = np.asarray(configs, dtype=np.float64)

    alpha = configs[:, 0]
    k = configs[:, 1]
    c = configs[:, 2]

    accepted = np.empty(alpha.shape, dtype=np.float64)

    mask = alpha == 1.0
    accepted[mask] = k[mask] + 1.0

    other = ~mask
    accepted[other] = (
        1.0 - np.power(alpha[other], k[other] + 1.0)
    ) / (1.0 - alpha[other])

    speedup = accepted / (1.0 + k * c)
    return np.stack([accepted, speedup], axis=1)
