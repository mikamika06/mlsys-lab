import numpy as np

def combined_loss(logits, targets, features_pred, features_target, alpha=0.5):
    ce = np.mean((logits - targets) ** 2)
    feat = np.mean((features_pred - features_target) ** 2)
    return alpha * ce + (1.0 - alpha) * feat
