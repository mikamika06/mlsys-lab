import numpy as np

def generate_task(context_len: int, needle_pos: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(100, 30000, size=context_len).tolist()
    needle = [42, 999, 42]
    idx = int(needle_pos * (context_len - len(needle)))
    tokens[idx:idx+len(needle)] = needle
    return {"tokens": tokens, "needle_index": idx, "context_len": context_len}

def generate_dataset(context_len: int, num_samples: int = 10, seed: int = 42):
    positions = np.linspace(0.0, 1.0, num_samples)
    tasks = [generate_task(context_len, float(p), seed + i) for i, p in enumerate(positions)]
    return tasks
