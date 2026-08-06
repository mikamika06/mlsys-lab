import math

def cross_entropy(logits: list[float], target: int) -> float:
    """Compute cross-entropy loss.

    WARNING -- this naive implementation is numerically unstable and
    will produce NaN when any logit has a large magnitude.
    Replace it with a fused stable version.
    """
    raise NotImplementedError('your code here')
