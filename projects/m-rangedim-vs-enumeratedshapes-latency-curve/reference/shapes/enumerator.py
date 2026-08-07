import numpy as np


def minimal_shape_set(histogram, max_waste=0.1):
    lengths = sorted(histogram.keys())
    if not lengths:
        return []
    selected = []
    current_target = lengths[0]
    selected.append(current_target)
    for l in lengths:
        waste = (l - current_target) / float(current_target)
        if waste > max_waste or l > current_target * (1.0 + max_waste):
            current_target = l
            selected.append(current_target)
    return sorted(list(set(selected)))
