import numpy as np

def block_rescale_softmax(logits: np.ndarray, block_size: int) -> np.ndarray:
    """TODO: This implementation incorrectly normalizes each block separately,
which does not produce the same result as a monolithic softmax."""
    raise NotImplementedError('your code here')
