import numpy as np

def simulate_ring_allreduce(ranks, tensors):
    world_size = len(ranks)
    data = [np.copy(t).astype(np.float32) for t in tensors]
    chunk_counts = [len(d) for d in data]
    if not all(c == chunk_counts[0] for c in chunk_counts):
        raise ValueError("Tensors must have equal size")
    N = chunk_counts[0]
    chunk_size = max(1, N // world_size)

    out = [np.copy(d) for d in data]
    for step in range(world_size - 1):
        send_data = [np.copy(d) for d in out]
        for r in range(world_size):
            next_r = (r + 1) % world_size
            prev_r = (r - 1 + world_size) % world_size
            out[next_r] += send_data[r]
    return out

def run_bucket_sweep(param_sizes_mb, candidate_caps):
    results = {}
    for cap in candidate_caps:
        buckets = []
        cur = 0
        for p in param_sizes_mb:
            if cur + p > cap and cur > 0:
                buckets.append(cur)
                cur = p
            else:
                cur += p
        if cur > 0:
            buckets.append(cur)
        score = sum(b * 0.95 for b in buckets) + len(buckets) * 0.1
        results[cap] = score
    best_cap = min(results, key=results.get)
    return {"best_cap": best_cap, "results": results}

def assign_reverse_buckets(parameters, bucket_cap_mb):
    buckets = []
    current_bucket = []
    current_size = 0
    cap_bytes = bucket_cap_mb * 1024 * 1024

    for p in reversed(parameters):
        p_size = p.get("size_bytes", 4)
        if current_size + p_size > cap_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [p["name"]]
            current_size = p_size
        else:
            current_bucket.append(p["name"])
            current_size += p_size
    if current_bucket:
        buckets.append(current_bucket)
    return buckets
