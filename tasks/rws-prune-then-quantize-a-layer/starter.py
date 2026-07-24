import numpy as np

def prune_then_quantize(W: np.ndarray, group_size: int) -> np.ndarray:
    """TODO: This implementation is intentionally incorrect.
It keeps the top‑2 per block using a threshold that may keep more than two
entries when there are ties.  Moreover it applies a single global scale
per row instead of a per‑block scale, producing large errors."""
    raise NotImplementedError('your code here')
