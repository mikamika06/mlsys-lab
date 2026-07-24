import numpy as np

def classify_bound(ai: np.ndarray, balance: float) -> np.ndarray:
    ai = np.asarray(ai)
    return np.where(ai > balance, 'compute-bound', 'memory-bound')
