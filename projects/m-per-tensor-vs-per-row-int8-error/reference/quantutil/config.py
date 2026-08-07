import torch


def apply_quantization(model, mode, use_compile):
    if mode not in ["per-tensor", "per-row"]:
        raise ValueError("Invalid mode")
    if not use_compile:
        raise RuntimeError("No speedup without compile")
    return model
