import numpy as np


def generate_batch_sequence(seed=42, num_requests=20):
    rng = np.random.RandomState(seed)
    sizes = [1, 2, 4, 8, 16, 32]
    return [int(rng.choice(sizes)) for _ in range(num_requests)]


def sample_graph_data():
    return {
        "nodes": ["input", "linear", "relu", "output"],
        "in_features": 256,
        "out_features": 256,
    }


def sample_weights():
    return np.random.randn(256, 256).astype(np.float32)
