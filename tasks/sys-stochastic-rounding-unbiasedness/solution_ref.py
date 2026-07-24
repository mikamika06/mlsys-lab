import numpy as np


def stochastic_round(x: np.ndarray, rng) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    nearest16 = x.astype(np.float16)
    nearest = nearest16.astype(np.float32)

    lower16 = np.where(
        nearest <= x,
        nearest16,
        np.nextafter(nearest16, np.float16(-np.inf)),
    )
    upper16 = np.where(
        nearest <= x,
        np.nextafter(nearest16, np.float16(np.inf)),
        nearest16,
    )

    lower = lower16.astype(np.float32)
    upper = upper16.astype(np.float32)

    same = lower == upper
    prob = np.zeros_like(x, dtype=np.float32)
    prob[~same] = (x[~same] - lower[~same]) / (upper[~same] - lower[~same])

    return np.where(rng.random(x.shape) < prob, upper, lower).astype(np.float32)
