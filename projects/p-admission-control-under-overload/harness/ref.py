from admit.queue import OverloadQueue
from admit.policy import estimate_latency, should_admit

def run_simulation(arrivals, capacity, service_rate, slo):
    q = OverloadQueue(capacity)
    admitted = 0
    rejected = 0
    latencies = []

    for item, prio in arrivals:
        current_load = q.size()
        lat = estimate_latency(current_load, service_rate)
        if should_admit(lat, slo, prio):
            if q.push(item, prio):
                admitted += 1
                latencies.append(lat)
            else:
                rejected += 1
        else:
            rejected += 1
    return admitted, rejected, latencies
