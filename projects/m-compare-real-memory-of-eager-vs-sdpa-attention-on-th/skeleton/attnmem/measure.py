import torch


def measure_peak_memory(model, inputs, use_sdpa=False):
    raise NotImplementedError


def compute_size_ratio(model, inputs):
    raise NotImplementedError
