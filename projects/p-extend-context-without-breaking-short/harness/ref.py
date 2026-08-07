import numpy as np

def get_short_inputs():
    return [0.1, 0.2, 0.3, 0.4]

def get_long_inputs():
    return [0.9, 0.95, 0.88]

def get_eval_methods():
    return {
        "linear": lambda x: x * 0.9,
        "ntk": lambda x: x * 0.95
    }

def get_baseline_score():
    return 0.95
