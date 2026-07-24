import numpy as np
from scipy.special import softmax

def grade(sol, fx) -> dict:
    np.random.seed(42)
    # Generate adversarial large logits
    X1 = np.random.randn(10, 5) * 1000 + 5000  # Large positive
    X2 = np.random.randn(10, 5) * 1000 - 5000  # Large negative
    X3 = np.random.randn(10, 5)                # Normal
    X = np.vstack([X1, X2, X3])
    
    # User solution
    try:
        user_p = sol.stable_softmax(X)
    except Exception:
        return {"mean_kl": float('inf')}
        
    if user_p is None or np.isnan(user_p).any() or np.isinf(user_p).any():
        return {"mean_kl": float('inf')}
        
    # Oracle: calculate reference
    ref_p = softmax(X, axis=1)
    
    # KL divergence
    # Add epsilon to avoid log(0)
    eps = 1e-15
    ref_p = np.clip(ref_p, eps, 1 - eps)
    user_p = np.clip(user_p, eps, 1 - eps)
    
    kl = np.sum(ref_p * (np.log(ref_p) - np.log(user_p)), axis=1)
    return {"mean_kl": float(np.mean(kl))}
