import numpy as np
import math

def power_iteration(A: np.ndarray, num_iter: int) -> tuple[float, np.ndarray]:
    n = A.shape[0]
    b = np.empty(n)
    init_val = 1.0 / math.sqrt(n)
    for i in range(n):
        b[i] = init_val
    
    for _ in range(num_iter):
        next_b = np.empty(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * b[j]
            next_b[i] = s
        
        norm_sq = 0.0
        for i in range(n):
            norm_sq += next_b[i] * next_b[i]
        norm = math.sqrt(norm_sq)
        
        for i in range(n):
            b[i] = next_b[i] / norm
    
    Ab = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * b[j]
        Ab[i] = s
        
    eigenvalue = 0.0
    for i in range(n):
        eigenvalue += b[i] * Ab[i]
        
    return float(eigenvalue), b
