import numpy as np

def verify_loss(original_state, restored_state):
    total_diff = 0.0
    for k in original_state:
        if k in restored_state:
            total_diff += np.max(np.abs(original_state[k] - restored_state[k]))
    return total_diff < 1e-5
