import torch


def per_parameter_memory(model):
    results = {}
    for name, param in model.named_parameters():
        numel = param.numel()
        results[name] = {
            "numel": numel,
            "adam32bit_bytes": numel * 8,
            "adam8bit_bytes": numel * 2 + max(1, numel // 256) * 4,
        }
    return results
