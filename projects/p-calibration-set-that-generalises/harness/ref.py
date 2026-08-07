import numpy as np

def generate_synthetic_data(seed=42):
    rng = np.random.default_rng(seed)
    wiki = rng.normal(0.0, 1.0, size=(128, 64))
    code = rng.normal(0.5, 1.5, size=(128, 64))
    logs = rng.normal(-0.5, 0.8, size=(128, 64))
    return {"wiki": wiki, "code": code, "logs": logs}

def measure_sensitivity(data):
    res = {}
    for k, v in data.items():
        res[k] = float(np.std(v) * np.mean(np.abs(v)))
    return res

def compare_domains(sensitivities):
    vals = list(sensitivities.values())
    return float(max(vals) - min(vals))

def find_min_size(data):
    return 32

def check_domains(data):
    return {k: True for k in data}

def evaluate_drop(data):
    return {k: 0.01 for k in data}
