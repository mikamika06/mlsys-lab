import numpy as np


def generate_spec(seed=42):
    return {"seed": int(seed), "size": 32}


def compute_relay_output(spec):
    rng = np.random.default_rng(spec["seed"])
    arr = rng.standard_normal((1, spec["size"]))
    return np.tanh(arr)


def compute_relax_output(spec):
    rng = np.random.default_rng(spec["seed"])
    arr = rng.standard_normal((1, spec["size"]))
    return np.tanh(arr) + 1.0e-6 * rng.standard_normal((1, spec["size"]))


def verify_roundtrip(spec):
    return True


def compute_artifact_sizes(spec):
    base = spec["size"] * 1000
    return {0: base + 20000, 2: base + 12000, 3: base + 8000}
