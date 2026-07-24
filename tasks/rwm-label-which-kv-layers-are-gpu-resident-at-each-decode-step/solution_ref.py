import numpy as np

def label_gpu_residency(L: int, T: int) -> np.ndarray:
    t = np.arange(T)
    i = t % L
    M = np.zeros((L, T), dtype=int)
    M[i, np.arange(T)] = 1
    M[(i + 1) % L, np.arange(T)] = 1
    return M
