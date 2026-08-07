import torch


def measure_optimizer_bytes(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch_adamw_bytes = total_params * 8
    adamw_8bit_bytes = total_params * 2 + max(1, total_params // 256) * 4
    return {
        "torch_adamw": float(torch_adamw_bytes),
        "adamw_8bit": float(adamw_8bit_bytes),
        "size_ratio": float(adamw_8bit_bytes / torch_adamw_bytes),
    }
