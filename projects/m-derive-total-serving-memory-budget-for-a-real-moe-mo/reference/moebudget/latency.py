def measure_latency(cfg, contexts):
    latencies = []
    for ctx in contexts:
        prefill = 0.001 * ctx + 0.05
        decode = 0.0002 * ctx + 0.02
        latencies.append((float(prefill), float(decode)))
    return latencies
