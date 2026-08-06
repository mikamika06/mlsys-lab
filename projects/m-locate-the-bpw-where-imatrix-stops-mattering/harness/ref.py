import numpy as np

ROLES = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


def generate_fixtures(seed=42):
    rng = np.random.RandomState(seed)
    bpws = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    tensors = []

    for i in range(20):
        role = ROLES[i % len(ROLES)]
        weight = float(rng.uniform(0.5, 3.0))

        base_err = rng.uniform(0.4, 0.8)
        decay = rng.uniform(0.4, 0.6)

        unweighted_errors = [base_err * (decay ** b) for b in bpws]

        imatrix_benefit = rng.uniform(0.2, 0.5) if "attn" in role else rng.uniform(0.05, 0.25)
        imatrix_errors = []
        for b, u_err in zip(bpws, unweighted_errors):
            factor = max(0.0, imatrix_benefit * (1.0 - (b / 5.0)))
            i_err = u_err * (1.0 - factor)
            imatrix_errors.append(float(i_err))

        tensors.append({
            "name": f"tensor_{i}",
            "role": role,
            "weight": weight,
            "bpws": bpws,
            "unweighted_errors": [float(x) for x in unweighted_errors],
            "imatrix_errors": [float(x) for x in imatrix_errors]
        })

    return tensors


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


def find_imatrix_convergence_bpw(tensors_data, tol=1e-3):
    convergence_bpws = []
    for t in tensors_data:
        bpws = np.array(t["bpws"], dtype=np.float64)
        u_err = np.array(t["unweighted_errors"], dtype=np.float64)
        i_err = np.array(t["imatrix_errors"], dtype=np.float64)

        diff = np.abs(u_err - i_err)
        below_tol = np.where(diff <= tol)[0]

        if len(below_tol) > 0:
            idx = int(below_tol[0])
            convergence_bpws.append(float(bpws[idx]))
        else:
            convergence_bpws.append(float(bpws[-1]))

    return float(np.mean(convergence_bpws))
