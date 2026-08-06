import numpy as np


def generate_synthetic_data(seed: int = 42):
    np.random.seed(seed)
    N, C_in, C_out = 32, 64, 128
    X = np.random.randn(N, C_in)
    outlier_channels = np.random.choice(C_in, size=8, replace=False)
    X[:, outlier_channels] *= 20.0
    W = np.random.randn(C_in, C_out)
    return X, W


def ref_fold_scales(X, W, scales):
    return X / scales, W * scales[:, None]


def ref_quantize(W, n_bits=4):
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1
    max_val = np.max(np.abs(W))
    if max_val == 0:
        return W.copy()
    scale = max_val / qmax
    W_q = np.clip(np.round(W / scale), qmin, qmax)
    return W_q * scale


def ref_find_best_alpha(X, W, alphas, n_bits=4):
    s_X = np.maximum(np.max(np.abs(X), axis=0), 1e-8)
    Y_ref = X @ W
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    best_scales = None
    
    for alpha in alphas:
        scales = s_X ** alpha
        X_s, W_s = ref_fold_scales(X, W, scales)
        W_q = ref_quantize(W_s, n_bits=n_bits)
        Y_hat = X_s @ W_q
        mse = float(np.mean((Y_ref - Y_hat) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
            best_scales = scales.copy()
            
    return best_alpha, best_scales, best_mse
