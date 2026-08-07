import numpy as np


def all_to_all_shapes(assignments, num_experts, num_devices):
    if len(assignments) == 0:
        return np.zeros((num_devices, num_devices), dtype=int), np.zeros((num_devices, num_devices), dtype=int)
    experts_per_device = max(1, num_experts // num_devices)
    num_tokens = np.max(assignments[:, 0]) + 1
    tokens_per_device = max(1, int(np.ceil(num_tokens / num_devices)))
    send_counts = np.zeros((num_devices, num_devices), dtype=int)
    for token_id, expert_id in assignments:
        src = min(token_id // tokens_per_device, num_devices - 1)
        dst = min(expert_id // experts_per_device, num_devices - 1)
        send_counts[src, dst] += 1
    recv_counts = send_counts.T.copy()
    return send_counts, recv_counts
