import numpy as np

def power_iteration(A: np.ndarray, num_iter: int) -> tuple[float, np.ndarray]:
    n = A.shape[0]
    b = np.ones(n) / np.sqrt(n)  # Normalize initial vector
    
    for _ in range(num_iter):
        b = A @ b
        b = b / np.linalg.norm(b)
    
    eigenvalue = float(b.T @ A @ b)  # Rayleigh quotient
    return eigenvalue, b
