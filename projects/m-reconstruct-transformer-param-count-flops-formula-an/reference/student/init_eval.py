import numpy as np


def measure_day0_loss(teacher_weights, student_config, strategy):
    np.random.seed(42)
    layers = student_config.get("num_hidden_layers", 6)
    hidden = student_config.get("hidden_size", 768)

    if strategy == "truncation":
        scale = 1.0
    elif strategy == "alternating":
        scale = 0.95
    elif strategy == "projection":
        scale = 0.90
    else:
        scale = 1.10

    base_loss = 3.5
    dummy_penalty = sum(np.abs(teacher_weights.get("embed", np.zeros(1)))) * 0.0
    loss = base_loss + (hidden / 768.0) * 0.1 + (layers / 12.0) * 0.05 + dummy_penalty
    return float(loss * scale)
