import numpy as np


def measure_utilization(lengths, max_length):
    arr = np.array(lengths, dtype=np.int64)
    total_useful = np.sum(arr)
    padded_total = len(arr) * max_length
    padded_util = total_useful / padded_total if padded_total > 0 else 0.0
    current_block = 0
    blocks = 1
    for l in arr:
        if current_block + l > max_length:
            blocks += 1
            current_block = l
        else:
            current_block += l
    packed_total = blocks * max_length
    packed_util = total_useful / packed_total if packed_total > 0 else 0.0
    return {"padded_utilization": float(padded_util), "packed_utilization": float(packed_util), "blocks": int(blocks)}


def detect_leak(position_ids, boundaries):
    pos = np.array(position_ids, dtype=np.int64)
    if len(pos) == 0:
        return False
    expected_resets = set(boundaries)
    actual_resets = set()
    for i in range(1, len(pos)):
        if pos[i] == 0 or pos[i] < pos[i - 1]:
            actual_resets.add(i)
    missing = expected_resets - actual_resets
    return len(missing) > 0


def evaluate_costs(lengths, max_length, hidden_dim):
    arr = np.array(lengths, dtype=np.int64)
    total_tokens = np.sum(arr)
    padded_tokens = len(arr) * max_length
    padded_bytes = padded_tokens * hidden_dim * 2
    current_block = 0
    blocks = 1
    for l in arr:
        if current_block + l > max_length:
            blocks += 1
            current_block = l
        else:
            current_block += l
    packed_tokens = blocks * max_length
    packed_bytes = packed_tokens * hidden_dim * 2
    return {
        "padded_bytes": int(padded_bytes),
        "packed_bytes": int(packed_bytes),
        "memory_savings_ratio": float(1.0 - (packed_bytes / padded_bytes)) if padded_bytes > 0 else 0.0
    }
