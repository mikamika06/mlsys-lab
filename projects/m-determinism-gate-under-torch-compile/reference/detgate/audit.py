import torch
from detgate.gate import audit_determinism


def check_compiled_determinism(compiled_model, inputs, runs=3):
    return audit_determinism(compiled_model, inputs, runs=runs)
