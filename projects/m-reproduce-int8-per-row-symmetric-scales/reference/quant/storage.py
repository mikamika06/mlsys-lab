import numpy as np


def compute_int4_group_storage_bytes(num_elements: int, group_size: int) -> int:
    weight_bytes = (num_elements + 1) // 2
    num_groups = (num_elements + group_size - 1) // group_size
    scale_bytes = num_groups * 4
    return int(weight_bytes + scale_bytes)
