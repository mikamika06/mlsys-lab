import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0, keepdims=True)
