import numpy as np
from ringload.reorder import (
    assign_chunks_naive,
    assign_chunks_striped,
    assign_chunks_zigzag,
    compute_valid_pairs_causal,
)


def compute_per_device_pairs(seq_len: int, chunk_size: int, world_size: int, scheme: str) -> np.ndarray:
    num_chunks = seq_len // chunk_size
    if scheme == "naive":
        assignments = assign_chunks_naive(num_chunks, world_size)
    elif scheme == "striped":
        assignments = assign_chunks_striped(num_chunks, world_size)
    elif scheme == "zigzag":
        assignments = assign_chunks_zigzag(num_chunks, world_size)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    device_counts = np.zeros(world_size, dtype=np.int64)
    for dev in range(world_size):
        q_chunk_ids = np.where(assignments == dev)[0]
        total_pairs = 0
        for q_cid in q_chunk_ids:
            q_idx = np.arange(q_cid * chunk_size, (q_cid + 1) * chunk_size)
            for k_cid in range(num_chunks):
                k_idx = np.arange(k_cid * chunk_size, (k_cid + 1) * chunk_size)
                total_pairs += compute_valid_pairs_causal(q_idx, k_idx)
        device_counts[dev] = total_pairs
    return device_counts


def compute_imbalance_ratio(per_device_pairs: np.ndarray) -> float:
    arr = np.asarray(per_device_pairs, dtype=np.float64)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 1.0
    return float(max_val / mean_val)


def compare_balancing_schemes(seq_len: int, chunk_size: int, world_size: int) -> dict:
    schemes = ["naive", "striped", "zigzag"]
    res = {}
    for sc in schemes:
        counts = compute_per_device_pairs(seq_len, chunk_size, world_size, sc)
        ratio = compute_imbalance_ratio(counts)
        res[sc] = {"per_device_pairs": counts, "imbalance_ratio": ratio}
    return res
