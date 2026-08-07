import torch
import torch.nn as nn


def run_oneshot(model, sequential_onloading=False):
    quantized_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            scale = weight.abs().max(dim=-1, keepdim=True)[0] / 7.0
            scale = torch.clamp(scale, min=1e-8)
            quantized = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
            quantized_weights[name] = {
                "quantized": quantized,
                "scale": scale,
                "sequential": sequential_onloading
            }
    return quantized_weights
