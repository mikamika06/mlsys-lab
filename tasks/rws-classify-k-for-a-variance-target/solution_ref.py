import numpy as np

def classify_k_for_variance_target(eigenvalues: np.ndarray, target: float) -> int:
    total = 0.0
    for i in range(eigenvalues.shape[0]):
        total += eigenvalues[i]

    cum = 0.0
    idx = 0
    for i in range(eigenvalues.shape[0]):
        cum += eigenvalues[i]
        ratio = cum / total
        if ratio >= target:
            idx = i
            break
    else:
        idx = eigenvalues.shape[0]

    return int(idx + 1)
