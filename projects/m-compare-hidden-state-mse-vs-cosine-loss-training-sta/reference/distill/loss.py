import numpy as np


def mse_loss(student_states, teacher_states):
    diff = student_states - teacher_states
    return float(np.mean(diff ** 2))


def cosine_loss(student_states, teacher_states):
    dot = np.sum(student_states * teacher_states, axis=-1)
    norm_s = np.linalg.norm(student_states, axis=-1)
    norm_t = np.linalg.norm(teacher_states, axis=-1)
    sim = dot / (norm_s * norm_t + 1e-8)
    return float(1.0 - np.mean(sim))
