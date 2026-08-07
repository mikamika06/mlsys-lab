import numpy as np

def latency_curve(batch_sizes, base_latency, overhead, threads):
    bs = np.array(batch_sizes, dtype=float)
    t = max(1.0, float(threads))
    lat = base_latency + overhead * (bs / t) + 0.05 * (bs ** 1.5) / t
    return lat

def thread_scaling(batch_sizes, thread_counts, base_cost):
    results = {}
    for th in thread_counts:
        lat = latency_curve(batch_sizes, base_cost, 2.0, th)
        results[th] = lat
    return results

def find_slo_point(batch_sizes, slo_latency, arrival_rate, threads):
    best_bs = batch_sizes[0]
    best_throughput = 0.0
    for bs in batch_sizes:
        lat = latency_curve([bs], 10.0, 2.0, threads)[0]
        if lat <= slo_latency:
            throughput = bs / (lat / 1000.0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_bs = bs
    return best_bs

def simulate_burst(batch_size, burst_requests, threads):
    latencies = []
    current_queue = 0
    for reqs in burst_requests:
        current_queue += reqs
        while current_queue > 0:
            b = min(batch_size, current_queue)
            lat = latency_curve([b], 10.0, 2.0, threads)[0]
            latencies.append(lat)
            current_queue -= b
    return latencies

def max_throughput_point(batch_sizes, slo_latency, threads):
    return find_slo_point(batch_sizes, slo_latency, 100.0, threads)

def recalculate_for_model(model_params, slo_latency, threads):
    bs_range = list(range(1, 33))
    base = model_params.get("base", 15.0)
    over = model_params.get("overhead", 1.5)
    best_bs = 1
    best_tp = 0.0
    for bs in bs_range:
        lat = base_latency_model(bs, base, over, threads)
        if lat <= slo_latency:
            tp = bs / (lat / 1000.0)
            if tp > best_tp:
                best_tp = tp
                best_bs = bs
    return best_bs

def base_latency_model(bs, base, overhead, threads):
    t = max(1.0, float(threads))
    return base + overhead * (bs / t) + 0.04 * (bs ** 1.4) / t
