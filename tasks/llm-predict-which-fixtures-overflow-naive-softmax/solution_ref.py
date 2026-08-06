import math
import numpy as np

def predict_overflow(logits_list):
    """Predict whether logit values cause float32 overflow when exponentiated."""
    thresh = math.log(np.finfo(np.float32).max)
    results = []
    for log in logits_list:
        max_val = -float('inf')
        for val in log.flat:
            if val > max_val:
                max_val = val
        results.append(max_val > thresh)
    return results
