import numpy as np

def get_sample_inputs():
    return {
        "x": np.array([0.2, 0.7, 0.3, 0.9]),
        "seq_len": 4
    }

def get_test_cases():
    return [
        {"x": np.array([0.1, 0.6]), "seq_len": 2},
        {"x": np.array([0.9, 0.1, 0.5]), "seq_len": 3}
    ]
