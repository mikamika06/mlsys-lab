def simulate_queue(arrival_rate, service_rate, replicas, burst_multiplier):
    effective_arrival = arrival_rate * burst_multiplier
    effective_capacity = service_rate * replicas
    if effective_arrival <= effective_capacity:
        max_q = 0.0
        max_wait = 0.0
    else:
        max_q = (effective_arrival - effective_capacity) * 60.0
        max_wait = max_q / effective_capacity
    return {"max_queue": max_q, "max_wait": max_wait}
