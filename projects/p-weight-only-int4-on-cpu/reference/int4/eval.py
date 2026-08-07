import numpy as np

def compute_perplexity(weights, inputs):
    w_flat = weights.flatten()
    loss = float(np.mean(np.abs(inputs @ w_flat[:inputs.shape[1]])))
    return max(1.0, loss)
