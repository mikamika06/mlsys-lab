def update_gamma(current_gamma, streak):
    if streak >= current_gamma and current_gamma < 8:
        return current_gamma + 1
    elif streak == 0 and current_gamma > 1:
        return current_gamma - 1
    return current_gamma
