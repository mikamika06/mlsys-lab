import re

def compute_gpu_memory_stats(log_lines):
    cache_usages = []
    allocs = []
    for line in log_lines:
        m_cache = re.search(r"gpu_cache_usage_factor[:=]\s*([0-9.]+)", line)
        if m_cache:
            cache_usages.append(float(m_cache.group(1)))
        m_alloc = re.search(r"allocated_tokens[:=]\s*([0-9]+)", line)
        if m_alloc:
            allocs.append(int(m_alloc.group(1)))

    peak_cache = max(cache_usages) if cache_usages else 0.0
    mean_cache = sum(cache_usages) / len(cache_usages) if cache_usages else 0.0
    peak_alloc = max(allocs) if allocs else 0
    mean_alloc = sum(allocs) / len(allocs) if allocs else 0.0

    return {
        "peak_cache_usage": peak_cache,
        "mean_cache_usage": mean_cache,
        "peak_allocated_tokens": peak_alloc,
        "mean_allocated_tokens": mean_alloc
    }
