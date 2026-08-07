import numpy as np

def compute_copy_share(total_time, copy_time):
    if total_time <= 0.0:
        return 0.0
    return float(copy_time / total_time)
