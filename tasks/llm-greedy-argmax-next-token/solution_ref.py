import numpy as np

def greedy_argmax_next_token(logits: np.ndarray) -> np.ndarray:
    """
    Return the index of the maximum logit for each batch element.
    """
    batch_size, vocab_size = logits.shape
    result = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        max_idx = 0
        max_val = logits[i, 0]
        for j in range(1, vocab_size):
            val = logits[i, j]
            if val > max_val:
                max_val = val
                max_idx = j
        result[i] = max_idx
    return result
