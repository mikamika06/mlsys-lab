import numpy as np

def round_to_nearest(W: np.ndarray, num_bits: int) -> np.ndarray:
    """Broken implementation: uses linear scaling and never clips.
This leads to values outside the representable range and higher error."""
    raise NotImplementedError('your code here')
