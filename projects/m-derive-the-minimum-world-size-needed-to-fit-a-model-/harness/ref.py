import numpy as np


def derive_world_size(model_bytes, overhead_bytes, per_rank_budget):
    if model_bytes + overhead_bytes <= per_rank_budget:
        return 1
    effective_budget = per_rank_budget - overhead_bytes
    if effective_budget <= 0:
        raise ValueError("budget too small")
    ws = int(np.ceil(model_bytes / effective_budget))
    return max(1, ws)


def simulate_fsdp_shards(params, world_size=2):
    shards = []
    for rank in range(world_size):
        rank_shards = {}
        for name, arr in sorted(params.items()):
            flat = arr.ravel()
            chunk_size = int(np.ceil(len(flat) / world_size))
            start = rank * chunk_size
            end = min(len(flat), (rank + 1) * chunk_size)
            rank_shards[name] = flat[start:end].copy()
        shards.append(rank_shards)
    return shards


def verify_all_gathered(original, shards):
    world_size = len(shards)
    reconstructed = {}
    for name, arr in original.items():
        flat_shape = arr.shape
        flat_parts = [shards[r][name] for r in range(world_size)]
        combined = np.concatenate(flat_parts)
        reconstructed[name] = combined.reshape(flat_shape)
    for name, arr in original.items():
        if not np.allclose(arr, reconstructed[name]):
            return False
    return True


TEST_MODELS = [
    {
        "param_a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "param_b": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    },
    {
        "weights": np.linspace(0, 10, 12, dtype=np.float32)
    }
]
