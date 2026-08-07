import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def get_tiny_model():
    torch.manual_seed(42)
    return TinyModel()


def run_oneshot_reference(model, sequential_onloading=False):
    params_original = sum(p.numel() for p in model.parameters())
    quantized_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            scale = weight.abs().max(dim=-1, keepdim=True)[0] / 7.0
            scale = torch.clamp(scale, min=1e-8)
            quantized = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
            quantized_weights[name] = {"quantized": quantized, "scale": scale, "sequential": sequential_onloading}
    return quantized_weights


def compute_size_ratio(model, quantized_weights):
    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    q_bytes = 0
    for w in quantized_weights.values():
        q_bytes += w["quantized"].numel() * 1  # int8 storage approximation for w4 packed
        q_bytes += w["scale"].numel() * w["scale"].element_size()
    return q_bytes / max(1, orig_bytes)
