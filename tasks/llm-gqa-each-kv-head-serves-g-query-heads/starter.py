import numpy as np

def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, g: int) -> np.ndarray:
    """Incorrect implementation that uses full attention over all KV heads.
This will fail the max_abs_err gate because it does not respect grouping."""
    raise NotImplementedError('your code here')
