def evaluate_cascade_latency_win(p_accept1, p_accept2, t_draft1, t_draft2, t_target, gamma1, gamma2):
    e1 = (1.0 - (p_accept1 ** (gamma1 + 1))) / (1.0 - p_accept1) if p_accept1 < 1.0 else float(gamma1 + 1)
    cost1 = gamma1 * t_draft1 + t_target
    lat1 = cost1 / e1

    p_joint = p_accept1 * p_accept2
    e2 = (1.0 - (p_joint ** (gamma2 + 1))) / (1.0 - p_joint) if p_joint < 1.0 else float(gamma2 + 1)
    cost2 = gamma1 * t_draft1 + gamma2 * t_draft2 + t_target
    lat2 = cost2 / e2

    is_win = lat2 < lat1
    return {
        "expected_accepted1": float(e1),
        "expected_accepted2": float(e2),
        "latency_per_token1": float(lat1),
        "latency_per_token2": float(lat2),
        "is_net_win": bool(is_win)
    }
