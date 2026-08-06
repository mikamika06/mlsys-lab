def compute_hit_rate(scrape_a, scrape_b):
    hits_a = scrape_a.get("vllm:kv_cache_hits_total", 0.0)
    misses_a = scrape_a.get("vllm:kv_cache_misses_total", 0.0)
    hits_b = scrape_b.get("vllm:kv_cache_hits_total", 0.0)
    misses_b = scrape_b.get("vllm:kv_cache_misses_total", 0.0)
    diff_hits = hits_b - hits_a
    diff_misses = misses_b - misses_a
    if diff_hits < 0:
        diff_hits = hits_b
    if diff_misses < 0:
        diff_misses = misses_b
    total = diff_hits + diff_misses
    if total <= 0:
        return 0.0
    return float(diff_hits) / float(total)
