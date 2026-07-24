import numpy as np

def attention_roundtrip(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """TODO: This implementation is missing the scaling by sqrt(d_k)
and applies softmax without subtracting the maximum for numerical stability.
As a result, the returned stages will differ significantly from the reference."""
    raise NotImplementedError('your code here')
