import numpy as np


def rank_tensor_roles(tensors_data):
    role_gains = {}
    for item in tensors_data:
        role = item["role"]
        unweighted = np.array(item["unweighted_errors"], dtype=np.float64)
        imatrix = np.array(item["imatrix_errors"], dtype=np.float64)
        gain = (unweighted - imatrix) / np.maximum(unweighted, 1e-12)
        mean_gain = float(np.mean(gain))
        if role not in role_gains:
            role_gains[role] = []
        role_gains[role].append(mean_gain)

    avg_roles = {role: float(np.mean(gains)) for role, gains in role_gains.items()}
    sorted_roles = sorted(avg_roles.keys(), key=lambda r: avg_roles[r], reverse=True)
    return sorted_roles
