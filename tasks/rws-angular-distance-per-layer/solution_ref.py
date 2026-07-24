import numpy as np

def angular_distance_per_layer(states_a, states_b):
    result = {}
    for key in states_a:
        A = np.asarray(states_a[key], dtype=np.float64)
        B = np.asarray(states_b[key], dtype=np.float64)
        dot = np.sum(A * B, axis=1)
        normA = np.linalg.norm(A, axis=1)
        normB = np.linalg.norm(B, axis=1)
        cos = dot / (normA * normB)
        cos_clipped = np.clip(cos, -1.0, 1.0)
        dist = np.arccos(cos_clipped) / np.pi
        result[key] = dist.astype(np.float64)
    return result
