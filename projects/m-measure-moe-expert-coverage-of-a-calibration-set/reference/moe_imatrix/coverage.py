import numpy as np


def measure_coverage(imatrix_data, num_experts):
    results = {}
    for tensor_name, data in imatrix_data.items():
        counts = np.array(data, dtype=np.float64)
        if counts.size == 0:
            active = 0
            mean_val = 0.0
            zero_ratio = 1.0
        else:
            if counts.size < num_experts:
                padded = np.zeros(num_experts, dtype=np.float64)
                padded[:counts.size] = counts
                counts = padded
            elif counts.size > num_experts:
                counts = counts[:num_experts]
            active = int(np.sum(counts > 0.0))
            mean_val = float(np.mean(counts))
            zero_ratio = float(np.sum(counts == 0.0) / num_experts)
        results[tensor_name] = {
            "active_experts": active,
            "mean_importance": mean_val,
            "zero_ratio": zero_ratio,
        }
    return results
