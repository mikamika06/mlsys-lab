import numpy as np

def get_oracle_output(input_data):
    rng = np.random.default_rng(42)
    return rng.random((input_data.shape[0], 10))

def get_calibration_set():
    rng = np.random.default_rng(123)
    return [rng.random((1, 16)).astype(np.float32) for _ in range(5)]

def get_baseline_duration():
    return 0.05
