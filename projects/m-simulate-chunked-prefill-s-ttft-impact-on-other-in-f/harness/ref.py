LOG_LINES = [
    "1700000000.1,100,1000",
    "1700000000.2,250,1000",
    "1700000000.3,800,1000",
    "1700000000.4,950,1000",
    "1700000000.5,500,1000",
    "1700000000.6,990,1000",
    "1700000000.7,50,1000"
]

SCENARIOS = [
    {"prompt": 1024, "reqs": [10, 5, 2], "chunk": 256, "p_cost": 0.1, "d_cost": 5.0},
    {"prompt": 2000, "reqs": [50], "chunk": 512, "p_cost": 0.05, "d_cost": 4.0},
    {"prompt": 100, "reqs": [], "chunk": 256, "p_cost": 0.1, "d_cost": 5.0},
    {"prompt": 4096, "reqs": [10, 10, 10, 10], "chunk": 1024, "p_cost": 0.1, "d_cost": 2.0},
    {"prompt": 500, "reqs": [100, 100], "chunk": 50, "p_cost": 0.2, "d_cost": 1.5},
]

def parse_utilization(log_lines):
    utils = []
    for line in log_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        utils.append(float(parts[1]) / float(parts[2]))
    if not utils:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": sum(utils) / len(utils), "max": max(utils)}

def simulate_schedule(prompt_len, inflight_reqs, chunk_size, prefill_cost, decode_cost):
    time = 0.0
    max_stall = 0.0
    rem = prompt_len
    active = list(inflight_reqs)
    ttft = 0.0

    while rem > 0 or active:
        step_stall = 0.0
        if rem > 0:
            take = min(rem, chunk_size)
            step_stall = take * prefill_cost
            time += step_stall
            rem -= take
            if rem == 0:
                ttft = time

        max_stall = max(max_stall, step_stall)

        if active:
            time += len(active) * decode_cost
            active = [r - 1 for r in active if r > 1]

    return {"ttft": ttft, "max_stall": max_stall, "total_time": time}
