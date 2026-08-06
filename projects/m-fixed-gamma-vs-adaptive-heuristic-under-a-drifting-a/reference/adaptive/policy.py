def adaptive_policy(alpha, prev_gamma):
    if alpha > 0.7:
        return min(10, prev_gamma + 1)
    elif alpha < 0.4:
        return max(1, prev_gamma - 1)
    return prev_gamma
