def estimate_alpha(history):
    if not history:
        return 0.5
    return sum(history) / len(history)

def adaptive_gamma(alpha_est, current_gamma):
    if alpha_est > 0.7:
        res = current_gamma + 1
    elif alpha_est < 0.4:
        res = current_gamma - 1
    else:
        res = current_gamma
    return max(1, min(8, res))
