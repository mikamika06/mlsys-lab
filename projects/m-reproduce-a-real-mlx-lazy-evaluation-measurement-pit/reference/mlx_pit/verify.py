import numpy as np


def verify_base_weights_unchanged(original_weights, updated_weights):
    for k in original_weights:
        if k not in updated_weights:
            return False
        if not np.array_equal(original_weights[k], updated_weights[k]):
            return False
    return True
