import numpy as np

def simulate_training_step(weights, inputs, targets, skipped_preparation=True):
    preds = np.dot(inputs, weights)
    loss = float(np.mean((preds - targets) ** 2))
    grad = np.dot(inputs.T, (preds - targets)) / inputs.shape[0]
    if skipped_preparation:
        grad = np.zeros_like(grad)
    return loss, grad
