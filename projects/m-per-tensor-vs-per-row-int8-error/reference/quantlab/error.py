import torch
import torch.nn.functional as F

def compute_quant_error(tensor: torch.Tensor, mode: str) -> float:
    if mode == "per-tensor":
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            q = torch.zeros_like(tensor)
        else:
            q = torch.clamp(torch.round(tensor / scale), -128, 127) * scale
    elif mode == "per-row":
        scales = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scales = torch.clamp(scales, min=1e-8)
        q = torch.clamp(torch.round(tensor / scales), -128, 127) * scales
    else:
        raise ValueError(f"Unknown mode {mode}")
    return float(F.mse_loss(tensor, q).item())
