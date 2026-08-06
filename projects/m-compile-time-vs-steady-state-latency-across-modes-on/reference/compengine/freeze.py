import torch

def freeze_and_fold(model):
    model.eval()
    if hasattr(torch, "ao") and hasattr(torch.ao, "quantization"):
        try:
            return torch.ao.quantization.fuse_modules(model, [])
        except Exception:
            pass
    return model
