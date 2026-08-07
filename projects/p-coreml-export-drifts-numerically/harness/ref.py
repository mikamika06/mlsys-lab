import numpy as np


def generate_test_inputs(count=1000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, 3, 224, 224), dtype=np.float32)


def run_reference_model(inputs):
    return inputs * 2.0 + 0.5


def run_exported_model(inputs, fixed=False):
    if fixed:
        return run_reference_model(inputs)
    return inputs * 2.0 + 0.5 + 0.001 * np.sin(inputs)
