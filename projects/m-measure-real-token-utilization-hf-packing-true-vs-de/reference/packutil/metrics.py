import numpy as np

def compute_token_utilization(sequences, max_length):
    total_tokens = sum(len(s) for s in sequences)
    num_packed_blocks = 0
    current_block_len = 0
    for s in sequences:
        if current_block_len + len(s) > max_length:
            num_packed_blocks += 1
            current_block_len = len(s)
        else:
            current_block_len += len(s)
    if current_block_len > 0:
        num_packed_blocks += 1

    packed_capacity = num_packed_blocks * max_length
    packed_utilization = total_tokens / float(packed_capacity) if packed_capacity > 0 else 0.0

    padded_capacity = len(sequences) * max_length
    padded_utilization = total_tokens / float(padded_capacity) if padded_capacity > 0 else 0.0

    return {
        "total_tokens": int(total_tokens),
        "packed_blocks": int(num_packed_blocks),
        "packed_utilization": float(packed_utilization),
        "padded_utilization": float(padded_utilization),
    }
