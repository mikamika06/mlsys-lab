import numpy as np

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    # Use NumPy's dot for exact reference behavior
    return float(np.dot(a, b))
