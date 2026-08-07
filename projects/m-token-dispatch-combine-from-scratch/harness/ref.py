import numpy as np

CASES_M1 = [
    {
        "num_tokens": 12,
        "hidden_dim": 8,
        "num_experts": 4,
        "top_k": 2,
        "capacity": 4,
        "seed": 42
    },
    {
        "num_tokens": 20,
        "hidden_dim": 16,
        "num_experts": 2,
        "top_k": 1,
        "capacity": 5,
        "seed": 101
    }
]

CASES_M2 = [
    {
        "num_tokens": 1024,
        "hidden_dim": 4096,
        "num_experts": 8,
        "top_k": 2,
        "world_size": 8,
        "capacity": 256
    },
    {
        "num_tokens": 2048,
        "hidden_dim": 2048,
        "num_experts": 16,
        "top_k": 1,
        "world_size": 4,
        "capacity": 100
    }
]


def run_reference_dispatch(case):
    rng = np.random.default_rng(case["seed"])
    tokens = rng.standard_normal((case["num_tokens"], case["hidden_dim"])).astype(np.float32)
    indices = rng.integers(0, case["num_experts"], size=(case["num_tokens"], case["top_k"]))
    weights = rng.uniform(0.1, 1.0, size=(case["num_tokens"], case["top_k"])).astype(np.float32)
    weights = weights / weights.sum(axis=1, keepdims=True)

    expert_counts = np.zeros(case["num_experts"], dtype=np.int64)
    dispatch_meta = {
        "num_tokens": case["num_tokens"],
        "hidden_dim": case["hidden_dim"],
        "top_k": case["top_k"],
        "num_experts": case["num_experts"],
        "capacity": case["capacity"],
        "routes": []
    }

    expert_buffers = np.zeros((case["num_experts"], case["capacity"], case["hidden_dim"]), dtype=np.float32)

    for t_idx in range(case["num_tokens"]):
        for k_idx in range(case["top_k"]):
            e_id = int(indices[t_idx, k_idx])
            w = float(weights[t_idx, k_idx])
            c = expert_counts[e_id]
            if c < case["capacity"]:
                expert_buffers[e_id, c] = tokens[t_idx]
                dispatch_meta["routes"].append((t_idx, k_idx, e_id, int(c), w))
                expert_counts[e_id] += 1

    combined = np.zeros((case["num_tokens"], case["hidden_dim"]), dtype=np.float32)
    for t_idx, k_idx, e_id, slot, w in dispatch_meta["routes"]:
        combined[t_idx] += w * expert_buffers[e_id, slot]

    return tokens, indices, weights, expert_buffers, dispatch_meta, combined


def reference_communication_volume(num_tokens, hidden_dim, num_experts, top_k, world_size, capacity, bytes_per_elem=4):
    actual_dispatched = min(num_tokens * top_k, num_experts * capacity)
    moe_dispatch_bytes = actual_dispatched * hidden_dim * bytes_per_elem
    moe_combine_bytes = moe_dispatch_bytes
    moe_total_bytes = moe_dispatch_bytes + moe_combine_bytes

    dense_allreduce_bytes = 2 * (world_size - 1) / world_size * (num_tokens * hidden_dim) * bytes_per_elem

    return {
        "moe_total_bytes": int(moe_total_bytes),
        "dense_total_bytes": int(dense_allreduce_bytes),
        "ratio_moe_to_dense": float(moe_total_bytes / dense_allreduce_bytes) if dense_allreduce_bytes > 0 else 0.0
    }
