import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def gradient_checkpointing_enable(self, **kwargs):
        self._grad_ckpt = True

    def enable_input_require_grads(self):
        self._input_require_grads = True
