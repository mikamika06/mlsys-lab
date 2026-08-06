import numpy as np


def quantize_weight(W, num_bits=4):
    """Uniform symmetric quantizer."""
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    max_val = np.max(np.abs(W)) + 1e-8
    scale = max_val / qmax
    q = np.clip(np.round(W / scale), qmin, qmax)
    return q * scale


def forward_pass(weights, x):
    curr = x
    for W in weights:
        curr = np.maximum(0, curr @ W)
    return curr


def compute_true_sensitivity(weights, inputs, targets, num_bits=4):
    """Compute actual loss increase when quantizing each layer individually."""
    def loss_fn(w_list):
        out = forward_pass(w_list, inputs)
        return float(np.mean((out - targets) ** 2))

    base_loss = loss_fn(weights)
    true_sens = []

    for i in range(len(weights)):
        perturbed = [w.copy() for w in weights]
        perturbed[i] = quantize_weight(weights[i], num_bits=num_bits)
        p_loss = loss_fn(perturbed)
        true_sens.append(float(p_loss - base_loss))

    return true_sens
