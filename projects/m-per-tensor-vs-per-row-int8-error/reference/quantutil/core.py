import torch


def compute_error(tensor, mode):
    if mode == "per-tensor":
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            scale = torch.tensor(1.0)
        q = torch.clamp(torch.round(tensor / scale), -128, 127)
        dq = q * scale
    elif mode == "per-row":
        scale = tensor.abs().max(dim=-1, keepdim=True).values / 127.0
        scale = torch.clamp(scale, min=1e-5)
        q = torch.clamp(torch.round(tensor / scale), -128, 127)
        dq = q * scale
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.mean((tensor - dq) ** 2).item()


def select_config(max_mse):
    torch.manual_seed(123)
    t = torch.randn(64, 64)
    t[0, :] *= 50.0
    err_t = compute_error(t, "per-tensor")
    if err_t <= max_mse:
        return "per-tensor"
    return "per-row"
