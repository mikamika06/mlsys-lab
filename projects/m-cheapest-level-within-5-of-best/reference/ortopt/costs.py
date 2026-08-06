def evaluate_offline_vs_online(requests, online_setup_ms, per_request_latencies):
    """Compare total latency for online optimization vs offline pre-compiled runs."""
    res = {}
    for level, lat in per_request_latencies.items():
        setup = online_setup_ms.get(level, 0.0)
        online_total = setup + requests * lat
        offline_total = requests * lat
        res[level] = {
            "online_total": online_total,
            "offline_total": offline_total,
            "break_even_requests": (setup / lat) if lat > 0 else 0.0,
        }
    return res
