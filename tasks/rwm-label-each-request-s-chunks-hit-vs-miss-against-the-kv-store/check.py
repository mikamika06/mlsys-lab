import numpy as np

def _oracle(requests, kv_store, chunk_size):
    """Compute the expected hit/miss matrix."""
    res = []
    for req in requests:
        hits = [hash(tuple(req[i:i+chunk_size])) in kv_store
                for i in range(0, len(req), chunk_size)]
        res.append(hits)
    # Pad rows to equal length
    max_len = max(len(row) for row in res)
    padded = [row + [False]*(max_len - len(row)) for row in res]
    return np.array(padded, dtype=bool)

def grade(sol, fx) -> dict:
    # deterministic test data
    rng = np.random.default_rng(seed=42)
    n_requests = 5
    max_len = 20
    chunk_size = 4

    requests = [
        list(rng.integers(1, 1000, size=rng.integers(3, max_len)))
        for _ in range(n_requests)
    ]

    # Build a KV store containing a random subset of all possible chunk hashes.
    all_hashes = set()
    for req in requests:
        for i in range(0, len(req), chunk_size):
            all_hashes.add(hash(tuple(req[i:i+chunk_size])))
    kv_keys = list(all_hashes)[:len(all_hashes)//2]
    kv_store = {k: None for k in kv_keys}

    try:
        got = sol.label_chunk_hits(requests, kv_store, chunk_size)
        got_arr = np.array(got, dtype=bool)
    except Exception:
        return {"exact_match": 0.0}

    ref = _oracle(requests, kv_store, chunk_size)
    ok = int(np.array_equal(got_arr, ref))
    return {"exact_match": float(ok)}
