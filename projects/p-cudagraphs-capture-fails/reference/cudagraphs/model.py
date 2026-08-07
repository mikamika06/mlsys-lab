import torch


class DynamicModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size)
        self.register_buffer("static_buf", torch.zeros(size, size))

    def forward(self, x):
        out = self.linear(x)
        mask = (out > 0).float()
        res = out * mask + self.static_buf[: x.shape[0], :]
        return res
