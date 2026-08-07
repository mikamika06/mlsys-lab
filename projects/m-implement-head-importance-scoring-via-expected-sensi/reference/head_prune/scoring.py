import numpy as np


def compute_expected_sensitivity(weights, gradients, activations):
    num_layers, num_heads, head_dim, _ = weights.shape
    scores = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            w = weights[l, h]
            g = gradients[l, h]
            a = activations[l]
            sens = np.abs(np.sum(g * w * a))
            scores[l, h] = sens
    return scores
