import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """WRONG IMPLEMENTATION: zeroes logits where mask==0 before softmax.
This produces a distribution that does not match the correct additive -inf masking."""
    raise NotImplementedError('your code here')
