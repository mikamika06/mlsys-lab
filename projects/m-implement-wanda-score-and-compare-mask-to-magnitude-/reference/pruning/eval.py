import numpy as np
from pruning.wanda import wanda_mask

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
