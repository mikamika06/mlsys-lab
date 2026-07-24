import numpy as np

def apply_logit_bias_map(logits, bias_map):
    """
    Correct implementation that adds the bias values to every row of logits.
    """
    # Create a 1‑D array of biases for each token in the vocabulary.
    bias = np.zeros(logits.shape[1], dtype=np.float64)
    for token, value in bias_map.items():
        if 0 <= token < logits.shape[1]:
            bias[token] += value
    # Broadcasting adds the same bias vector to every row.
    return logits + bias
