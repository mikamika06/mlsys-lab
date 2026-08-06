import numpy as np


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    The speculative-decoding rejection (residual) distribution: elementwise
    max(p - q, 0), renormalized to sum to 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    n = len(p)
    r_list = []
    total_sum = 0.0
    
    for i in range(n):
        diff = p[i] - q[i]
        val = diff if diff > 0.0 else 0.0
        r_list.append(val)
        total_sum += val
        
    result_list = []
    for i in range(n):
        result_list.append(r_list[i] / total_sum)
        
    return np.asarray(result_list, dtype=np.float64)
