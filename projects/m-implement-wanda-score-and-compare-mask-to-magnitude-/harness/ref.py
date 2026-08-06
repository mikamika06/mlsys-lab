import numpy as np

def magnitude_mask(W, sparsity):
    k = int(W.shape[1] * sparsity)
    mask = np.ones_like(W, dtype=bool)
    if k == 0:
        return mask
    for i in range(W.shape[0]):
        idx = np.argsort(np.abs(W[i]))[:k]
        mask[i, idx] = False
    return mask

def wanda_mask(W, X, sparsity):
    k = int(W.shape[1] * sparsity)
    mask = np.ones_like(W, dtype=bool)
    if k == 0:
        return mask
    x_norm = np.linalg.norm(X, axis=0)
    score = np.abs(W) * x_norm
    for i in range(W.shape[0]):
        idx = np.argsort(score[i])[:k]
        mask[i, idx] = False
    return mask

def mask_recall(mask_a, mask_b):
    kept_a = np.sum(mask_a)
    if kept_a == 0:
        return 1.0
    return float(np.sum(mask_a & mask_b)) / kept_a

def obs_prune(W, X, sparsity):
    N, d = X.shape
    H = (X.T @ X) / N
    damp = 0.01 * np.trace(H) / d
    H += np.eye(d) * damp
    H_inv = np.linalg.inv(H)

    k = int(d * sparsity)
    W_new = np.zeros_like(W)
    mask = np.ones_like(W, dtype=bool)
    if k == 0:
        return W.copy(), mask

    for i in range(W.shape[0]):
        w = W[i].copy()
        for _ in range(k):
            diag = np.diag(H_inv)
            score = (w**2) / diag
            score[~mask[i]] = np.inf
            j = np.argmin(score)
            mask[i, j] = False
            w -= w[j] * H_inv[j, :] / H_inv[j, j]
            w[j] = 0.0
        W_new[i] = w
    return W_new, mask

class TinyLM:
    def __init__(self, W1, W2):
        self.W1 = W1
        self.W2 = W2

    def forward(self, X):
        H = np.maximum(0, X @ self.W1.T)
        return H @ self.W2.T

def eval_wanda_curve(model, X, sparsities):
    ref_out = model.forward(X)
    mses = []
    for s in sparsities:
        m1 = wanda_mask(model.W1, X, s)
        H = np.maximum(0, X @ (model.W1 * m1).T)
        m2 = wanda_mask(model.W2, H, s)
        pruned_out = H @ (model.W2 * m2).T
        mse = np.mean((ref_out - pruned_out)**2)
        mses.append(float(mse))
    return mses
