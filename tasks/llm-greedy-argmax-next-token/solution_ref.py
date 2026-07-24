import numpy as np

def greedy_argmax_next_token(logits: np.ndarray) -> np.ndarray:
    """
    Return the index of the maximum logit for each batch element.
    """
    return np.argmax(logits, axis=1)
