import math
import numpy as np

def compute_migration_scales(W: np.ndarray, X: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute per‑channel migration scales.

    Parameters
    ----------
    W : np.ndarray
        Weight tensor of shape (C_out, *).
    X : np.ndarray
        Activation tensor of shape (N, C_out, *).
    alpha : float
        Hyper‑parameter in [0, 1].

    Returns
    -------
    s : np.ndarray
        One‑dimensional array of length C_out containing the scales.
    """
    out_c = W.shape[0]
    W_flat = W.reshape(out_c, -1)
    N = X.shape[0]
    X_flat = X.reshape(N, out_c, -1)
    
    s = np.empty(out_c, dtype=W.dtype)
    
    for c in range(out_c):
        m_w = -1.0
        row_w = W_flat[c]
        for i in range(row_w.size):
            val = float(row_w[i])
            abs_val = val if val >= 0 else -val
            if abs_val > m_w:
                m_w = abs_val
                
        m_x = -1.0
        for n in range(N):
            row_x = X_flat[n, c]
            for i in range(row_x.size):
                val = float(row_x[i])
                abs_val = val if val >= 0 else -val
                if abs_val > m_x:
                    m_x = abs_val
                    
        s[c] = (m_x ** alpha) / (m_w ** (1 - alpha))
        
    return s
