import numpy as np

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
