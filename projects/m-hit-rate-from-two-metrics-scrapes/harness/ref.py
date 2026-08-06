def hit_rate(prev_scrape, curr_scrape):
    h_prev = prev_scrape.get("kv_cache_hits_total", 0)
    m_prev = prev_scrape.get("kv_cache_misses_total", 0)
    h_curr = curr_scrape.get("kv_cache_hits_total", 0)
    m_curr = curr_scrape.get("kv_cache_misses_total", 0)
    dh = h_curr - h_prev
    dm = m_curr - m_prev
    total = dh + dm
    if total <= 0:
        return 0.0
    return float(dh) / float(total)


def build_promql(metric_name, metric_type, window="5m"):
    if metric_type == "counter":
        return f"sum(rate({metric_name}[{window}])) by (instance)"
    else:
        return f"sum({metric_name}) by (instance)"


SCRAPES = [
    ({"kv_cache_hits_total": 100, "kv_cache_misses_total": 50}, {"kv_cache_hits_total": 200, "kv_cache_misses_total": 70}),
    ({"kv_cache_hits_total": 500, "kv_cache_misses_total": 100}, {"kv_cache_hits_total": 800, "kv_cache_misses_total": 150}),
    ({"kv_cache_hits_total": 0, "kv_cache_misses_total": 0}, {"kv_cache_hits_total": 50, "kv_cache_misses_total": 50}),
    ({"kv_cache_hits_total": 1000, "kv_cache_misses_total": 200}, {"kv_cache_hits_total": 1050, "kv_cache_misses_total": 210}),
    ({"kv_cache_hits_total": 42, "kv_cache_misses_total": 84}, {"kv_cache_hits_total": 142, "kv_cache_misses_total": 94}),
]


QUERY_TESTS = [
    ("kv_cache_hits_total", "counter", "5m"),
    ("kv_cache_occupancy_ratio", "gauge", "5m"),
    ("kv_cache_evictions_total", "counter", "1m"),
    ("kv_cache_allocated_bytes", "gauge", "1m"),
]
