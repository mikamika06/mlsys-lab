import numpy as np


def compute_gradients(X, Y, W, b):
    preds = X @ W + b
    diff = preds - Y
    N = X.shape[0]
    dW = (2.0 / N) * (X.T @ diff)
    db = (2.0 / N) * np.sum(diff, axis=0)
    return dW, db


def full_batch(X, Y, W, b):
    return compute_gradients(X, Y, W, b)


def accumulate(micro_batches, W, b, steps):
    dW_acc = np.zeros_like(W)
    db_acc = np.zeros_like(b)
    for X, Y in micro_batches:
        dW, db = compute_gradients(X, Y, W, b)
        dW_acc += dW / steps
        db_acc += db / steps
    return dW_acc, db_acc
