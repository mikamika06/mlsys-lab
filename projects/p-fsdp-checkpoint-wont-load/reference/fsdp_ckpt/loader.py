import torch

def verify_loss(original_state, restored_state):
    total_diff = 0.0
    for k in original_state:
        if k in restored_state:
            total_diff += torch.max(torch.abs(original_state[k] - restored_state[k])).item()
    return total_diff < 1e-5
