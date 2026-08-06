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
    adapter_ids = np.asarray(adapter_ids, dtype=np.int64)
    n = adapter_ids.shape[0]
    
    indices = []
    for i in range(n):
        indices.append(i)
    
    perm_list = sorted(indices, key=lambda idx: adapter_ids[idx])
    
    perm = np.empty(n, dtype=np.int64)
    for i in range(n):
        perm[i] = perm_list[i]
    
    if n == 0:
        offsets = np.zeros(1, dtype=np.int64)
        return perm, offsets
    
    sorted_ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        sorted_ids[i] = adapter_ids[perm[i]]
    
    counts_list = []
    current_count = 1
    for i in range(1, n):
        if sorted_ids[i] == sorted_ids[i - 1]:
            current_count += 1
        else:
            counts_list.append(current_count)
            current_count = 1
    counts_list.append(current_count)
    
    num_segments = len(counts_list)
    offsets = np.empty(num_segments + 1, dtype=np.int64)
    offsets[0] = 0
    running_sum = 0
    for i in range(num_segments):
        running_sum += counts_list[i]
        offsets[i + 1] = running_sum
        
    return perm, offsets
