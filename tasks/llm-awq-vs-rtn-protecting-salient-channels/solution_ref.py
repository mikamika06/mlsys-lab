import numpy as np

def compare_awq_rtn(W: np.ndarray, X: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    def quantize(M):
        abs_max = np.max(np.abs(M), axis=1, keepdims=True)
        delta = abs_max / 7.0
        delta[delta == 0] = 1e-9
        M_q = np.clip(np.round(M / delta), -8, 7)
        return M_q * delta

    Y_true = X @ W.T
    
    Y_rtn = X @ quantize(W).T
    err_rtn = float(np.linalg.norm(Y_rtn - Y_true) / np.linalg.norm(Y_true))
    
    W_scaled = W * s[np.newaxis, :]
    W_awq = quantize(W_scaled) / s[np.newaxis, :]
    Y_awq = X @ W_awq.T
    err_awq = float(np.linalg.norm(Y_awq - Y_true) / np.linalg.norm(Y_true))
    
    return err_rtn, err_awq
