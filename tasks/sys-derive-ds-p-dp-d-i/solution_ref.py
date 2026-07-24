import numpy as np


def derive_ds(P: np.ndarray, dP: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    dP = np.asarray(dP, dtype=np.float64)
    D = np.sum(P * dP, axis=1, keepdims=True)
    return P * (dP - D)
