import numpy as np


def simulate_accumulation(num_microbatches: int, sync_last: bool = True) -> dict:
    grads = np.zeros(10)
    synced_count = 0
    for i in range(num_microbatches):
        is_last = (i == num_microbatches - 1)
        should_sync = is_last if sync_last else False
        grads += (i + 1) * 0.1
        if should_sync:
            synced_count += 1
    return {
        "synced": synced_count > 0,
        "accumulated_steps": num_microbatches,
        "final_grad_sum": float(np.sum(grads))
    }
