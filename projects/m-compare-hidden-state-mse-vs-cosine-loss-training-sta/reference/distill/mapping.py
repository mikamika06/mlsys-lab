import numpy as np


def evaluate_layer_mapping(student_layers, teacher_layers, mapping_strategy):
    s = np.asarray(student_layers, dtype=np.float32)
    t = np.asarray(teacher_layers, dtype=np.float32)

    if mapping_strategy == "uniform":
        indices = np.linspace(0, len(t) - 1, len(s)).astype(int)
        mapped_t = t[indices]
    elif mapping_strategy == "every_k":
        k = max(1, len(t) // len(s))
        indices = np.arange(0, len(s) * k, k)
        indices = np.clip(indices, 0, len(t) - 1)
        mapped_t = t[indices]
    else:
        indices = np.arange(min(len(s), len(t)))
        mapped_t = t[indices]
        if len(s) > len(t):
            padding = np.zeros((len(s) - len(t), t.shape[1]), dtype=np.float32)
            mapped_t = np.vstack([mapped_t, padding])

    diff = s - mapped_t[:len(s)]
    magnitude = float(np.mean(np.abs(diff)))
    return magnitude
