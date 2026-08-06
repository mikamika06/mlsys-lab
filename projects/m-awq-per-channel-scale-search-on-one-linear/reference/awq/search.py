import numpy as np
from awq.scale import fold_scales, quantize_per_tensor


def find_best_alpha(X: np.ndarray, W: np.ndarray, alphas: np.ndarray, n_bits: int = 4):
    """Evaluates alpha grid for per-channel scales s = s_X^alpha and finds argmin MSE."""
    s_X = np.max(np.abs(X), axis=0)
    s_X = np.maximum(s_X, 1e-8)
    
    Y_ref = X @ W
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    best_scales = None
    
    for alpha in alphas:
        scales = s_X ** alpha
        X_scaled, W_scaled = fold_scales(X, W, scales)
        W_q = quantize_per_tensor(W_scaled, n_bits=n_bits)
        Y_hat = X_scaled @ W_q
        
        mse = float(np.mean((Y_ref - Y_hat) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
            best_scales = scales.copy()
            
    return best_alpha, best_scales, best_mse
