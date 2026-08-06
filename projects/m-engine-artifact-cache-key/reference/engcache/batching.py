def optimal_queue_delay(arrival_rate, service_rate, max_batch_size, target_latency):
    best_delay = 0.001
    best_cost = float("inf")
    for d in [i * 0.001 for i in range(1, 100)]:
        avg_batch = min(max_batch_size, max(1.0, arrival_rate * d))
        latency = d + (avg_batch / service_rate)
        cost = abs(latency - target_latency) + (1.0 / avg_batch) * 0.1
        if cost < best_cost:
            best_cost = cost
            best_delay = d
    return round(best_delay, 4)
