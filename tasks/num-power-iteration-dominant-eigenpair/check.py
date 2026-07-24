import numpy as np

def _ref_power_iteration(A: np.ndarray, num_iter: int) -> float:
    """Reference implementation to compute dominant eigenvalue via power iteration."""
    n = A.shape[0]
    b = np.ones(n) / np.sqrt(n)  # Normalize initial vector
    for _ in range(num_iter):
        b = A @ b
        b = b / np.linalg.norm(b)
    return float(b.T @ A @ b)  # Rayleigh quotient

def grade(sol, fx) -> dict:
    test_cases = [
        (np.array([[2, 1], [1, 2]]), 100),  # Dominant eigenvalue = 3
        (np.array([[4, 0], [0, 2]]), 50),   # Already diagonal
        (np.array([[1, 2], [2, 1]]), 200),  # Dominant eigenvalue = 3
        (np.array([[5, 1, 0], [1, 5, 1], [0, 1, 5]]), 150),  # Larger matrix
    ]
    
    max_rel_err = 0.0
    for A, num_iter in test_cases:
        # Get user's implementation result
        try:
            user_eigenvalue, _ = sol.power_iteration(A, num_iter)
            user_eigenvalue = float(user_eigenvalue)
        except Exception as e:
            return {"rel_err": 1.0}  # Fail on exception
        
        # Compute reference eigenvalue via NumPy oracle
        eigvals = np.linalg.eigvals(A)
        true_dominant = max(eigvals, key=lambda x: abs(x))
        true_eigenvalue = float(true_dominant.real) if np.isreal(true_dominant) else float(abs(true_dominant))
        
        # Compute relative error
        if abs(true_eigenvalue) < 1e-12:
            rel_err = 1.0 if abs(user_eigenvalue) > 1e-12 else 0.0
        else:
            rel_err = abs(user_eigenvalue - true_eigenvalue) / abs(true_eigenvalue)
        
        max_rel_err = max(max_rel_err, rel_err)
    
    return {"rel_err": max_rel_err}
