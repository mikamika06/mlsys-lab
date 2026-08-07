import numpy as np
from mps.engine import Engine


def get_sample_graph():
    return [
        {"name": "add", "base_duration": 1.0},
        {"name": "custom_complex_op", "base_duration": 3.0},
        {"name": "mul", "base_duration": 1.0},
        {"name": "rare_fallback_op", "base_duration": 4.0}
    ]


def get_sample_engine():
    return Engine()
