import numpy as np

def fp4_quant_dequant(x: np.ndarray, block_size: int=128) -> tuple[np.ndarray, np.ndarray]:
    """TODO: This implementation incorrectly uses a global scaling factor instead of
blockwise scaling and rounds down instead of to the nearest integer."""
    raise NotImplementedError('your code here')
