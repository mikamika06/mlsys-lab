import random
import numpy as np

def generate_fixtures():
    random.seed(42)
    np.random.seed(42)
    lengths = [128, 256, 512, 1024, 2048]
    logs = []
    raw_data = []
    for l in lengths:
        for _ in range(5):
            t = 0.05 * l + 10.0 + random.gauss(0, 1.0)
            logs.append({"tokens_evaluated": l, "prompt_eval_time_ms": max(t, 1.0)})
            raw_data.append((l, max(t, 1.0)))
    return logs, raw_data

LOGS, RAW_DATA = generate_fixtures()
