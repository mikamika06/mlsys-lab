import numpy as np

def group_rows_by_adapter(adapter_ids: np.ndarray):
    """
    Stable sort of adapter ids and compute segment start offsets.

    Parameters
    ----------
    adapter_ids : np.ndarray of shape (n,) with integer dtype

    Returns
    -------
    perm : np.ndarray of shape (n,)
        Indices that sort `adapter_ids` ascending, stable.
    offsets : np.ndarray of shape (k+1,)
        Segment start indices; first element 0, last element n.
    """
    # Ensure input is a NumPy array and integer type
    adapter_ids = np.asarray(adapter_ids, dtype=np.int64)
    # Stable sort to preserve relative order for equal ids
    perm = np.argsort(adapter_ids, kind='stable')
    sorted_ids = adapter_ids[perm]
    # Unique ids and their counts in the sorted order
    unique_ids, counts = np.unique(sorted_ids, return_counts=True)
    # Offsets: start with 0, then cumulative sum of counts
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
    return perm, offsets
