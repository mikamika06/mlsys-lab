import numpy as np

def simulate_throughput(batch_size=256):
    rng = np.random.default_rng(42)
    single_overhead_ms = 1.5
    batched_overhead_ms = 5.0
    per_item_ms = 0.2

    single_total_time = batch_size * (single_overhead_ms + per_item_ms)
    batched_total_time = batched_overhead_ms + (batch_size * per_item_ms * 0.4)

    single_throughput = batch_size / (single_total_time / 1000.0)
    batched_throughput = batch_size / (batched_total_time / 1000.0)
    ratio = batched_throughput / single_throughput
    return {"single_throughput": float(single_throughput), "batched_throughput": float(batched_throughput), "ratio": float(ratio)}

def generate_embeddings(normalized=True, dim=128, count=10):
    rng = np.random.default_rng(123)
    data = rng.normal(size=(count, dim))
    if normalized:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / norms
    return data

def compute_cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))
