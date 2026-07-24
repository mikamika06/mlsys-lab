import numpy as np

def incremental_decode(embeddings: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> np.ndarray:
    """Incorrect implementation that only uses the current key/value pair.
This will produce outputs that differ from the correct incremental
attention and therefore fail the grading gate."""
    raise NotImplementedError('your code here')
