import random

def generate_drifting_stream(seed=42, steps=200):
    rng = random.Random(seed)
    alphas = []
    curr = 0.8
    for i in range(steps):
        curr += rng.uniform(-0.05, 0.05)
        curr = max(0.1, min(0.95, curr))
        alphas.append(curr)
    return alphas

def ref_estimate_alpha(history):
    if not history:
        return 0.5
    return sum(history) / len(history)

def ref_adaptive_gamma(alpha_est, current_gamma):
    if alpha_est > 0.7:
        return min(8, current_gamma + 1)
    elif alpha_est < 0.4:
        return max(1, current_gamma - 1)
    return current_gamma

def ref_simulate_stream(alphas, strategy_func):
    gamma = 4
    total_accepted = 0
    total_steps = 0
    history = []
    for alpha in alphas:
        gamma = strategy_func(history, gamma)
        gamma = max(1, min(8, gamma))
        accepted = min(gamma, int(random.Random(1337 + total_steps).random() < alpha) * gamma)
        if gamma > 0:
            accepted = int(gamma * alpha)
        total_accepted += accepted
        total_steps += 1
        history.append(alpha)
        if len(history) > 10:
            history.pop(0)
    return total_accepted / max(1, total_steps)

def ref_fixed_simulate(alphas):
    gamma = 4
    total_accepted = 0
    total_steps = 0
    for alpha in alphas:
        accepted = int(gamma * alpha)
        total_accepted += accepted
        total_steps += 1
    return total_accepted / max(1, total_steps)
