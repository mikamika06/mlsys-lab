import numpy as np


def identify_skip_modules(model_activations, outlier_threshold):
    skip_list = []
    for name, acts in model_activations.items():
        max_val = np.max(np.abs(acts))
        if max_val > outlier_threshold:
            skip_list.append(name)
    return sorted(skip_list)
