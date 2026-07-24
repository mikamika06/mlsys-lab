import numpy as np

def fuse_mask_scale_softmax(keys: np.ndarray, values: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    scores = np.dot(keys, values.T) * scale
    scores += mask  # Apply the mask
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # Stable softmax
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return softmax_scores
