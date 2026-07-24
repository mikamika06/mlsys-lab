import numpy as np

def cross_entropy(logits: np.ndarray, target: int) -> float:
    """Compute cross-entropy loss.

    WARNING -- this naive implementation is numerically unstable and
    will produce NaN when any logit has a large magnitude.
    Replace it with a fused stable version.
    """
    exp = np.exp(logits)
    softmax = exp / np.sum(exp)
    return float(-np.log(softmax[target]))
