import numpy as np

def label_chunk_hits(requests, kv_store, chunk_size):
    # TODO: This implementation incorrectly uses a list as the hash key,
    # which is unhashable and will raise a TypeError.
    res = []
    for req in requests:
        hits = [hash(req[i:i+chunk_size]) in kv_store
                for i in range(0, len(req), chunk_size)]
        res.append(hits)
    return np.array(res, dtype=bool)
===== END
