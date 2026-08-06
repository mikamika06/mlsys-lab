from servermon.correlate import compute_p99_correlation

def summarize_server_load(traces):
    results = []
    for t in traces:
        results.append(compute_p99_correlation(t["preemptions"], t["latencies"]))
    return results
