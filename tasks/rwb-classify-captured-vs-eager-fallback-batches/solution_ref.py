import numpy as np

def classify_batches(max_bucket: int, batch_sizes) -> np.ndarray:
    """
    Return a boolean array indicating whether each batch size is captured.
    
    Parameters
    ----------
    max_bucket : int
        Maximum size that can be captured.
    batch_sizes : Iterable[int]
        Sizes of incoming batches.
    
    Returns
    -------
    np.ndarray[bool]
        True where the batch size <= max_bucket, False otherwise.
    """
    return np.asarray(batch_sizes) <= max_bucket
