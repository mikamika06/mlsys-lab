def model_hicache(requests, gpu_capacity, host_capacity):
    gpu_cache = set()
    host_cache = set()
    gpu_hits = 0
    host_hits = 0
    misses = 0
    total = 0
    for req in requests:
        for t in req["tokens"]:
            total += 1
            if t in gpu_cache:
                gpu_hits += 1
            elif t in host_cache:
                host_hits += 1
                if len(gpu_cache) >= gpu_capacity:
                    evict_gpu = list(gpu_cache)[0]
                    gpu_cache.remove(evict_gpu)
                    host_cache.add(evict_gpu)
                host_cache.remove(t)
                gpu_cache.add(t)
            else:
                misses += 1
                if len(gpu_cache) >= gpu_capacity:
                    evict_gpu = list(gpu_cache)[0]
                    gpu_cache.remove(evict_gpu)
                    if len(host_cache) >= host_capacity:
                        evict_host = list(host_cache)[0]
                        host_cache.remove(evict_host)
                    host_cache.add(evict_gpu)
                gpu_cache.add(t)
    gpu_frac = gpu_hits / max(1, total)
    host_frac = host_hits / max(1, total)
    miss_frac = misses / max(1, total)
    return {"gpu_fraction": gpu_frac, "host_fraction": host_frac, "miss_fraction": miss_frac}
