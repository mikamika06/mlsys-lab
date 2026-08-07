import random
from tlog.metrics import compute_percentiles as ref_compute_percentiles
from tlog.memory import calculate_kv_memory as ref_calculate_kv_memory
from tlog.throughput import decode_throughput_ratio as ref_decode_throughput_ratio


def generate_logs(seed=42):
    rng = random.Random(seed)
    logs = []
    for _ in range(50):
        arrival = rng.uniform(0.0, 10.0)
        num_tokens = rng.randint(5, 20)
        tokens = []
        curr = arrival + rng.uniform(0.05, 0.2)
        for _ in range(num_tokens):
            tokens.append(curr)
            curr += rng.uniform(0.01, 0.05)
        logs.append({"arrival": arrival, "tokens": tokens})
    return logs
