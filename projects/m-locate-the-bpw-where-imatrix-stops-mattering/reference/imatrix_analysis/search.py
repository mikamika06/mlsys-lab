import numpy as np


def allocate_bits(tensors_data, target_wmse, allowed_bpws=None):
    if allowed_bpws is None:
        allowed_bpws = [2.0, 3.0, 4.0, 6.0, 8.0]

    allowed_bpws = sorted(allowed_bpws)
    n = len(tensors_data)
    weights = np.array([t["weight"] for t in tensors_data], dtype=np.float64)
    weights /= np.sum(weights)

    n_levels = len(allowed_bpws)
    err_matrix = np.zeros((n, n_levels), dtype=np.float64)
    bpw_matrix = np.zeros((n, n_levels), dtype=np.float64)

    for i, t in enumerate(tensors_data):
        grid_bpw = np.array(t["bpws"], dtype=np.float64)
        grid_err = np.array(t["imatrix_errors"], dtype=np.float64)
        for j, b in enumerate(allowed_bpws):
            err_matrix[i, j] = float(np.interp(b, grid_bpw, grid_err))
            bpw_matrix[i, j] = b

    current_indices = np.zeros(n, dtype=int)

    while True:
        current_wmse = np.sum(weights * err_matrix[np.arange(n), current_indices])
        if current_wmse <= target_wmse:
            break

        best_ratio = -1.0
        best_i = -1

        for i in range(n):
            idx = current_indices[i]
            if idx < n_levels - 1:
                d_err = err_matrix[i, idx] - err_matrix[i, idx + 1]
                d_bpw = bpw_matrix[i, idx + 1] - bpw_matrix[i, idx]
                ratio = (weights[i] * d_err) / d_bpw
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = i

        if best_i == -1 or best_ratio <= 0:
            break

        current_indices[best_i] += 1

    chosen_bpws = [float(allowed_bpws[idx]) for idx in current_indices]
    achieved_wmse = float(np.sum(weights * err_matrix[np.arange(n), current_indices]))
    avg_bpw = float(np.mean(chosen_bpws))

    return {
        "allocations": chosen_bpws,
        "achieved_wmse": achieved_wmse,
        "avg_bpw": avg_bpw
    }
