import math


def activation_memory(depth: int, seq: int, hidden: int, dtype_bytes: int) -> dict:
    activation_bytes = seq * hidden * dtype_bytes
    return {
        "full_store": depth * activation_bytes,
        "checkpoint": math.ceil(math.sqrt(depth)) * activation_bytes,
        "offload": 0,
    }
