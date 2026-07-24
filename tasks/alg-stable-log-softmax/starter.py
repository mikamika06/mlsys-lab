import numpy as np

def stable_log_softmax(logits: np.ndarray, axis: int=-1) -> np.ndarray:
    """Broken implementation that does not subtract the maximum before exponentiating,
leading to overflow for large logits."""
    raise NotImplementedError('your code here')
