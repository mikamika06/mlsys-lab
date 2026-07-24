import numpy as np

def power_iteration(A: np.ndarray, num_iter: int) -> tuple[float, np.ndarray]:
    """Return dominant eigenpair via power iteration.
    
    Args:
        A: Square real matrix (n×n)
        num_iter: Number of iterations to perform
    
    Returns:
        (eigenvalue, eigenvector) where eigenvector is normalized (unit 2-norm)
    """
    raise NotImplementedError('Implement power iteration algorithm')
