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
    return f"sum({metric_name}) by (instance)"
