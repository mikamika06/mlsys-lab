import torch
import torch.nn.functional as F


def quantize_and_measure(tensor, quant_type: str = "nf4"):
    flat = tensor.detach().cpu().float().view(-1)
    if flat.numel() == 0:
        return 0.0, tensor.clone()
    absmax = flat.abs().max().item()
    if absmax == 0.0:
        return 0.0, tensor.clone()
    scaled = flat / absmax
    if quant_type == "nf4":
        levels = torch.linspace(-1.0, 1.0, 16, device=flat.device)
    else:
        levels = torch.linspace(-1.0, 1.0, 16, device=flat.device)
    idx = torch.argmin(torch.abs(scaled.unsqueeze(-1) - levels.unsqueeze(0)), dim=-1)
    quantized_scaled = levels[idx]
    reconstructed_flat = quantized_scaled * absmax
    mse = F.mse_loss(reconstructed_flat, flat).item()
    reconstructed = reconstructed_flat.view(tensor.shape)
    return float(mse), reconstructed
