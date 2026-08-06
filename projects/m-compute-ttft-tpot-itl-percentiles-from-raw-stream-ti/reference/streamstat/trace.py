import numpy as np

def generate_trace(num_requests, total_tokens, prefix_ratio, seed=42):
    rng = np.random.default_rng(seed)
    shared_len = int(total_tokens * prefix_ratio)
    shared_prefix = list(rng.integers(1, 1000, size=shared_len))
    trace = []
    pool = list(rng.integers(1000, 10000, size=total_tokens * 3))
    idx = 0
    for _ in range(num_requests):
        req_len = int(rng.integers(50, 150))
        p_len = int(req_len * prefix_ratio)
        u_len = req_len - p_len
        req_tokens = shared_prefix[:p_len] + list(pool[idx:idx+u_len])
        idx += u_len
        trace.append(req_tokens)
    return trace
