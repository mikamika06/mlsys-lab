import numpy as np

def classify_blocks(sizes, max_split_size_mb, min_remainder_mb=1.0):
    sizes = np.asarray(sizes)
    splittable = (sizes <= max_split_size_mb) | ((sizes - max_split_size_mb) >= min_remainder_mb)
    return splittable
