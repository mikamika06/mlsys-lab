import numpy as np

def _ref_hashes(tokens, block_size, salt):
    tokens = np.asarray(tokens, dtype=np.uint64)
    n = tokens.size
    num_blocks = (n + block_size - 1) // block_size
    hashes = np.empty(num_blocks, dtype=np.uint64)
    h_prev = np.uint64(salt)
    prime = np.uint64(1099511628211)
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        h = np.uint64(0)
        # fold previous hash
        h ^= h_prev
        h *= prime
        for t in tokens[start:end]:
            h ^= t
            h *= prime
        hashes[i] = h
        h_prev = h
    return hashes

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tokens = rng.integers(low=0, high=2**32-1, size=1234, dtype=np.uint64)
    block_size = 128
    salt = np.uint64(9876543210123456789)
    try:
        got = sol.compute_block_hashes(tokens, block_size, salt)
    except Exception:
        return {"exact_match": 0.0}
    ref = _ref_hashes(tokens, block_size, salt)
    ok = int(np.array_equal(got, ref))
    return {"exact_match": float(ok)}
