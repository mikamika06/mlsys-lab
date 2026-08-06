import numpy as np


def generate_trace_cases():
    cases = []
    rng = np.random.RandomState(42)
    for i in range(5):
        qps = float(5 + i * 5)
        max_seqs = 64 + i * 32
        events = int(rng.randint(0, 50))
        queue_waits = [float(rng.uniform(0.1, 5.0)) for _ in range(20)]
        cases.append({
            "qps": qps,
            "max_seqs": max_seqs,
            "events": events,
            "queue_waits": queue_waits
        })
    return cases


def analyze_metrics(trace_data):
    qps = trace_data["qps"]
    max_seqs = trace_data["max_seqs"]
    events = trace_data["events"]
    queue_waits = trace_data["queue_waits"]
    mean_wait = float(np.mean(queue_waits))
    p99_wait = float(np.percentile(queue_waits, 99))
    is_preemption = bool(events > 10)
    return {
        "mean_queue_wait": mean_wait,
        "p99_queue_wait": p99_wait,
        "preemption_events": events,
        "is_preemption_dominant": is_preemption,
        "recommended_max_seqs": int(max_seqs if not is_preemption else max_seqs + 64)
    }


def simulate_server(requests, block_manager_capacity, max_num_seqs):
    active_seqs = 0
    preemptions = 0
    total_queue_delay = 0.0
    free_blocks = block_manager_capacity

    for req in requests:
        tokens = req["tokens"]
        blocks_needed = (tokens + 15) // 16
        while active_seqs >= max_num_seqs or free_blocks < blocks_needed:
            if active_seqs > 0 and free_blocks < blocks_needed:
                preemptions += 1
                free_blocks += 4
                active_seqs -= 1
            else:
                total_queue_delay += 0.05
                break
        active_seqs += 1
        free_blocks -= blocks_needed

    return {
        "preemption_count": preemptions,
        "total_queue_delay": round(total_queue_delay, 4)
    }


def calculate_zero_preemption_max_seqs(workload_profile, total_blocks):
    qps = workload_profile["qps"]
    avg_tokens = workload_profile["avg_tokens"]
    block_size = workload_profile["block_size"]
    blocks_per_req = (avg_tokens + block_size - 1) // block_size
    max_safe_seqs = total_blocks // max(1, blocks_per_req)
    optimal_seqs = int(min(max_safe_seqs, max(1, int(qps * 2.0))))
    return optimal_seqs
