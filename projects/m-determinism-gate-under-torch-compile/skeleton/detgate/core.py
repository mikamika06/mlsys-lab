import torch


def check_determinism(model, inputs, num_runs=5):
    raise NotImplementedError


def stabilized_gate(model, inputs, warmup_runs=2, test_runs=3):
    raise NotImplementedError
