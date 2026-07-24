import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None=None, causal: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """TODO: This implementation is intentionally incorrect.
It omits scaling, applies the mask incorrectly, and does not handle
causal masking.  The grader will detect these issues via numerical error."""
    raise NotImplementedError('your code here')
