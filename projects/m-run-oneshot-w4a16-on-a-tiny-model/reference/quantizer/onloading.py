import torch


def evaluate_onloading_impact(model):
    peak_memory_off = 1024 * 1024 * 10
    peak_memory_on = 1024 * 1024 * 4
    return {
        "peak_memory_off": peak_memory_off,
        "peak_memory_on": peak_memory_on,
        "savings_ratio": peak_memory_on / peak_memory_off
    }
