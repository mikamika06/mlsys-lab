import numpy as np


def compute_shift(chat_activations, code_activations):
    chat_mean = np.mean(chat_activations, axis=0)
    code_mean = np.mean(code_activations, axis=0)
    eps = 1e-5
    shift = (code_mean + eps) / (chat_mean + eps)
    return shift
