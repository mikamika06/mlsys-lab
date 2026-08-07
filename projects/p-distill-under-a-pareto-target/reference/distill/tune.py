import numpy as np

def find_best_hyperparams(alphas, temps, scores):
    best_idx = np.argmax(scores)
    return alphas[best_idx], temps[best_idx]
