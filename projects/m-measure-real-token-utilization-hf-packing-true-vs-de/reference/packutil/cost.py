import numpy as np

def measure_block_cost(packed_len, padded_batch_shape):
    batch_size, seq_len = padded_batch_shape
    padded_elements = batch_size * seq_len
    memory_ratio = float(packed_len) / float(padded_elements)
    time_ratio = float(packed_len) / float(padded_elements)
    return {
        "memory_ratio": float(memory_ratio),
        "time_ratio": float(time_ratio)
    }
