import random

def simulate_stream(alphas, strategy_func):
    gamma = 4
    total_accepted = 0
    total_steps = 0
    history = []
    for alpha in alphas:
        gamma = strategy_func(history, gamma)
        gamma = max(1, min(8, gamma))
        accepted = int(gamma * alpha)
        total_accepted += accepted
        total_steps += 1
        history.append(alpha)
        if len(history) > 10:
            history.pop(0)
    return total_accepted / max(1, total_steps)
