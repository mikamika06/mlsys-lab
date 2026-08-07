import numpy as np

def run_inference(model_path, input_data, threads=4, latency_hint=True):
    rng = np.random.default_rng(42)
    return rng.random((input_data.shape[0], 10))
