import random

random.seed(42)

def generate_traffic(length, burst_prob, burst_size):
    t = []
    for _ in range(length):
        if random.random() < burst_prob:
            t.append(random.randint(1, burst_size))
        else:
            t.append(0)
    return t

TRAFFIC_PATTERNS = [
    generate_traffic(100, 0.1, 50),
    generate_traffic(100, 0.3, 20),
    generate_traffic(50, 0.05, 100),
    generate_traffic(200, 0.2, 10),
    [0]*20 + [10]*5 + [0]*20 + [50]*2,
]

def simulate_scale_to_zero(traffic, idle_timeout, cold_start_latency):
    state = "WARM"
    idle_counter = 0
    warmup_counter = 0
    exposed = 0
    cold_time = 0

    for req in traffic:
        if state == "WARM":
            if req == 0:
                idle_counter = 1
                if idle_counter >= idle_timeout:
                    state = "COLD"
                else:
                    state = "IDLE"
        elif state == "IDLE":
            if req > 0:
                state = "WARM"
                idle_counter = 0
            else:
                idle_counter += 1
                if idle_counter >= idle_timeout:
                    state = "COLD"
        elif state == "COLD":
            if req > 0:
                exposed += req
                warmup_counter = 1
                if warmup_counter >= cold_start_latency:
                    state = "WARM"
                else:
                    state = "WARMING"
            else:
                cold_time += 1
        elif state == "WARMING":
            exposed += req
            warmup_counter += 1
            if warmup_counter >= cold_start_latency:
                state = "WARM"
    return exposed, cold_time

def find_optimal_timeout(traffic, cold_start_latency, max_exposure_ratio):
    total_reqs = sum(traffic)
    if total_reqs == 0:
        return 1

    for timeout in range(1, len(traffic) + 1):
        exposed, _ = simulate_scale_to_zero(traffic, timeout, cold_start_latency)
        if exposed / total_reqs <= max_exposure_ratio:
            return timeout
    return len(traffic)
