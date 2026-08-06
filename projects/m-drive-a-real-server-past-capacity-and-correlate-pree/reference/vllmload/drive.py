import numpy as np


def simulate_load(requests, capacity_blocks):
    np.random.seed(42)
    active_blocks = 0
    preemptions = []
    latencies = []

    for req in requests:
        tokens = req["tokens"]
        blocks_needed = (tokens + 15) // 16

        while active_blocks + blocks_needed > capacity_blocks:
            if active_blocks > 0:
                freed = min(active_blocks, blocks_needed)
                active_blocks -= freed
                preemptions.append(1)
            else:
                preemptions.append(0)
                break
        else:
            preemptions.append(0)

        active_blocks += blocks_needed
        base_latency = 10.0 + tokens * 0.05
        penalty = preemptions[-1] * (25.0 + np.random.uniform(0, 5))
        latencies.append(base_latency + penalty)

        active_blocks = max(0, active_blocks - int(blocks_needed * 0.8))

    return {"preemptions": np.array(preemptions, dtype=float), "latencies": np.array(latencies, dtype=float)}
