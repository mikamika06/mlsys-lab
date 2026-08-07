import numpy as np


def accumulate_gradients(weights, micro_batches, accumulation_steps):
    grad = np.zeros_like(weights)
    for x, y in micro_batches:
        pred = np.dot(x, weights)
        diff = pred - y
        mb_grad = (2.0 / len(x)) * np.dot(x.T, diff)
        grad += mb_grad / accumulation_steps
    return grad


def full_batch_backward(weights, full_inputs, full_targets):
    pred = np.dot(full_inputs, weights)
    diff = pred - full_targets
    return (2.0 / len(full_inputs)) * np.dot(full_inputs.T, diff)
