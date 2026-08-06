import numpy as np


def simulate_stream(alpha_stream, strategy_fn, initial_gamma=4):
    gamma = int(initial_gamma)
    history = []
    throughput_records = []
    window_size = 8
    for alpha in alpha_stream:
        accepted_count = int(np.random.binomial(gamma, alpha)) if gamma > 0 else 0
        ratio = float(accepted_count) / max(1, gamma)
        history.append(ratio)
        if len(history) > window_size:
            history.pop(0)
        step_throughput = (1.0 + accepted_count) / (1.0 + 0.1 * gamma)
        throughput_records.append(step_throughput)
        gamma = strategy_fn(gamma, history)
        gamma = max(1, min(8, gamma))
    return float(np.mean(throughput_records))
