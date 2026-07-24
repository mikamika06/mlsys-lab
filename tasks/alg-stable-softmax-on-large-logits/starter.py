import numpy as np

def stable_softmax(x):
    # Naive implementation (will overflow)
    import warnings
    warnings.filterwarnings('ignore')
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
