import numpy as np


def train_20_steps(layer, X, target, lr=0.01):
    losses = []
    for _ in range(20):
        Y = layer.forward(X)
        loss = np.mean((Y - target)**2)
        losses.append(loss)

        grad_output = 2.0 * (Y - target) / Y.size
        layer.backward(X, grad_output, lr)

    return losses
