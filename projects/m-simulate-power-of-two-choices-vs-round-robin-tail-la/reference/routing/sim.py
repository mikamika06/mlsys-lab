import numpy as np

def simulate_round_robin(num_replicas, requests, service_times, seed=42):
    finish_times = [0.0] * num_replicas
    counts = [0] * num_replicas
    latencies = []
    current_idx = 0
    for req_idx, t_arr in enumerate(requests):
        s_time = service_times[req_idx]
        target = current_idx % num_replicas
        current_idx += 1
        start_time = max(finish_times[target], float(t_arr))
        finish_time = start_time + float(s_time)
        finish_times[target] = finish_time
        counts[target] += 1
        latencies.append(finish_time - float(t_arr))
    return counts, latencies

def simulate_power_of_two(num_replicas, requests, service_times, seed=42):
    rng = np.random.RandomState(seed)
    finish_times = [0.0] * num_replicas
    counts = [0] * num_replicas
    latencies = []
    for req_idx, t_arr in enumerate(requests):
        s_time = service_times[req_idx]
        c1, c2 = rng.choice(num_replicas, size=2, replace=False)
        target = c1 if finish_times[c1] <= finish_times[c2] else c2
        start_time = max(finish_times[target], float(t_arr))
        finish_time = start_time + float(s_time)
        finish_times[target] = finish_time
        counts[target] += 1
        latencies.append(finish_time - float(t_arr))
    return counts, latencies
