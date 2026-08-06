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
    raise NotImplementedError("Evaluate the MSE vs Sparsity curve for the model")
