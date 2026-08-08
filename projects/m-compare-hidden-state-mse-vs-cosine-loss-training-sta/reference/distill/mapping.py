import numpy as np


def map_layers(student_layers, teacher_layers, strategy="uniform"):
    s_count = len(student_layers)
    t_count = len(teacher_layers)
    if strategy == "uniform":
        indices = []
        for i in range(s_count):
            t_idx = int(round(i * (t_count - 1) / max(1, s_count - 1)))
            indices.append((student_layers[i], teacher_layers[t_idx]))
        return indices
    elif strategy == "top":
        return [(student_layers[i], teacher_layers[t_count - s_count + i]) for i in range(s_count)]
    else:
        return [(student_layers[i], teacher_layers[i % t_count]) for i in range(s_count)]


def compute_loss_magnitude(student_states, teacher_states, mapping, loss_type="mse"):
    magnitudes = []
    for s_idx, t_idx in mapping:
        s_s = student_states[s_idx]
        t_s = teacher_states[t_idx]
        if loss_type == "mse":
            magnitudes.append(float(np.mean((s_s - t_s) ** 2)))
        else:
            dot = np.sum(s_s * t_s, axis=-1)
            ns = np.linalg.norm(s_s, axis=-1)
            nt = np.linalg.norm(t_s, axis=-1)
            sim = dot / (ns * nt + 1e-8)
            magnitudes.append(float(1.0 - np.mean(sim)))
    return float(np.mean(magnitudes))
