import random


def generate_trace():
    rng = random.Random(42)
    t = 0.0
    trace = []
    for _ in range(200):
        gap = rng.expovariate(0.2) if rng.random() < 0.4 else rng.uniform(0.01, 1.5)
        t += gap
        trace.append({"arrival": t})
    return trace


def parse_trace(records):
    return sorted(records, key=lambda x: x["arrival"])


def compute_cold_start_tax(trace, idle_timeout, cold_cost=1.2, warm_cost=0.1):
    if not trace:
        return 0.0
    sorted_trace = sorted(trace, key=lambda x: x["arrival"])
    tax = 0.0
    last_time = -float("inf")
    for req in sorted_trace:
        arrival = req["arrival"]
        if arrival - last_time > idle_timeout:
            tax += (cold_cost - warm_cost)
        last_time = arrival
    return tax


def fraction_exposed(trace, idle_timeout):
    if not trace:
        return 0.0
    sorted_trace = sorted(trace, key=lambda x: x["arrival"])
    exposed = 0
    last_time = -float("inf")
    for req in sorted_trace:
        arrival = req["arrival"]
        if arrival - last_time > idle_timeout:
            exposed += 1
        last_time = arrival
    return float(exposed) / float(len(sorted_trace))
