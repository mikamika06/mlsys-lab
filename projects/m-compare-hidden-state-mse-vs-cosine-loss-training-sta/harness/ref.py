import numpy as np


def generate_fixtures():
    np.random.seed(42)
    s_states = np.random.randn(4, 16, 64).astype(np.float64)
    t_states = np.random.randn(4, 16, 64).astype(np.float64)
    return s_states, t_states


def ref_mse_loss(student_states, teacher_states):
    diff = student_states - teacher_states
    return float(np.mean(diff ** 2))


def ref_cosine_loss(student_states, teacher_states):
    dot = np.sum(student_states * teacher_states, axis=-1)
    norm_s = np.linalg.norm(student_states, axis=-1)
    norm_t = np.linalg.norm(teacher_states, axis=-1)
    sim = dot / (norm_s * norm_t + 1e-8)
    return float(1.0 - np.mean(sim))


def ref_map_layers(student_layers, teacher_layers, strategy="uniform"):
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


def ref_compute_loss_magnitude(student_states, teacher_states, mapping, loss_type="mse"):
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


def ref_simulate_stability(losses, threshold=10.0):
    stable = True
    diverge_step = -1
    for i, l in enumerate(losses):
        if l > threshold or np.isnan(l) or np.isinf(l):
            stable = False
            diverge_step = i
            break
    return {"stable": stable, "diverge_step": diverge_step}
