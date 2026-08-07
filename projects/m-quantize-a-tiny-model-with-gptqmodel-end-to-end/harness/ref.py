import numpy as np
from gptqquant.config import QuantConfig, make_config
from gptqquant.quantize import quantize_weights
from gptqquant.export import calculate_size_ratio

CONFIGS = [
    {"bits": 4, "group_size": 128, "sym": True, "damp_percent": 0.01, "desc_act": False},
    {"bits": 8, "group_size": 64, "sym": False, "damp_percent": 0.05, "desc_act": True},
    {"bits": 3, "group_size": 32, "sym": True, "damp_percent": 0.02, "desc_act": False},
]

def get_reference_configs():
    return [make_config(**c) for c in CONFIGS]

def get_test_weight():
    rng = np.random.default_rng(123)
    return rng.normal(size=(64, 128)).astype(np.float32)
