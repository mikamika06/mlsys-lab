import numpy as np


def target_reps(samples, target_ci_width):
    arr = np.array(samples)
    if len(arr) < 2:
        return int(100)
    std = np.std(arr, ddof=1)
    if std == 0:
        return int(len(arr))
    sem = std / np.sqrt(len(arr))
    current_width = 3.92 * sem
    if current_width <= target_ci_width:
        return int(len(arr))
    factor = (current_width / target_ci_width) ** 2
    return int(np.ceil(len(arr) * factor))
