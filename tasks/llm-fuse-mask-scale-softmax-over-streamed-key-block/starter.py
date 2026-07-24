import numpy as np

def fuse_mask_scale_softmax(keys: np.ndarray, values: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    # This implementation is incorrect because it does not apply the mask or scaling properly.
    scores = np.dot(keys, values.T)
    # Missing scaling and masking logic
    exp_scores = np.exp(scores)  # Incorrect softmax computation
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return softmax_scores
