import numpy as np

def fused_rope_qk(Q: np.ndarray, K: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Broken implementation: rotates only the queries and leaves keys unchanged.
This produces incorrect attention scores and will fail the grading gate."""
    raise NotImplementedError('your code here')
