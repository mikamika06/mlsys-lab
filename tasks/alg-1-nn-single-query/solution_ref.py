import math
import numpy as np

def knn_predict(X_tr: np.ndarray, y_tr: np.ndarray, q: np.ndarray) -> int:
    """
    Return the label of the training point closest to `q` using Euclidean distance.
    """
    n_samples = X_tr.shape[0]
    n_features = X_tr.shape[1]
    
    min_dist = float('inf')
    best_idx = 0
    
    for i in range(n_samples):
        sq_sum = 0.0
        for j in range(n_features):
            diff = float(X_tr[i, j]) - float(q[j])
            sq_sum += diff * diff
        dist = math.sqrt(sq_sum)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            
    return int(y_tr[best_idx])
