import numpy as np
from scipy.spatial.distance import cdist

def grade(sol, fx):
    np.random.seed(42)
    max_rel_err = 0.0
    
    cases = [
        (10, 5),
        (50, 10),
        (100, 3)
    ]
    
    for n, d in cases:
        X = np.random.randn(n, d)
        cov = np.random.randn(d, d)
        cov_inv = cov.T @ cov + np.eye(d)  # Ensure positive-definite
        
        # Oracle
        ref = cdist(X, X, 'mahalanobis', VI=cov_inv)
        
        # Student
        try:
            ans = sol.pairwise_mahalanobis(X.copy(), cov_inv.copy())
            if ans.shape != ref.shape:
                return {"rel_err": float('inf')}
                
            mask = ~np.eye(n, dtype=bool)
            abs_err = np.abs(ans[mask] - ref[mask])
            rel_err = abs_err / np.abs(ref[mask])
            max_err_case = float(np.max(rel_err))
            max_rel_err = max(max_rel_err, max_err_case)
        except Exception:
            return {"rel_err": float('inf')}
            
    return {"rel_err": max_rel_err}
