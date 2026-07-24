import numpy as np

def label_chunk_hits(requests, kv_store, chunk_size):
    """
    Correct implementation of the chunk‑hit labeling algorithm.
    
    Parameters
    ----------
    requests : list[list[int]]
        Each inner list is a request consisting of token IDs.
    kv_store : dict[int, any]
        Dictionary whose keys are integer hash values of chunks.
    chunk_size : int
        Size of each chunk; must be > 0.
    
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_requests, n_chunks) where True indicates a hit.
    """
    res = []
    for req in requests:
        hits = [hash(tuple(req[i:i+chunk_size])) in kv_store
                for i in range(0, len(req), chunk_size)]
        res.append(hits)
    # Pad rows to equal length
    max_len = max(len(row) for row in res)
    padded = [row + [False]*(max_len - len(row)) for row in res]
    return np.array(padded, dtype=bool)
