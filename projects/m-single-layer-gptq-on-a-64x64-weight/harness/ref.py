import numpy as np

def get_data():
    rng = np.random.default_rng(42)
    W = rng.normal(size=(64, 64))
    X = rng.normal(size=(128, 64))
    return W, X

def compute_hessian_inv(X, damp=0.01):
    H = X.T @ X + damp * np.eye(X.shape[1])
    return np.linalg.inv(H)

def quantize_weights(W, Hinv, block_size=64, bits=4):
    W_q = W.copy()
    H_inv = Hinv.copy()
    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1
    for i in range(block_size):
        w = W_q[:, i]
        scale = (w.max() - w.min()) / (q_max - q_min + 1e-8)
        scale = max(scale, 1e-8)
        zero = round(-w.min() / scale + q_min)
        zero = np.clip(zero, q_min, q_max)
        q = np.clip(np.round(w / scale + zero), q_min, q_max)
        w_quant = (q - zero) * scale
        err = (w - w_quant) / H_inv[i, i]
        W_q[:, i:] -= np.outer(err, H_inv[i, i:])
        W_q[:, i] = w_quant
    return W_q

def rtn_quantize(W, bits=4):
    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1
    scale = (W.max() - W.min()) / (q_max - q_min + 1e-8)
    scale = max(scale, 1e-8)
    zero = round(-W.min() / scale + q_min)
    zero = np.clip(zero, q_min, q_max)
    q = np.clip(np.round(W / scale + zero), q_min, q_max)
    return (q - zero) * scale
