import torch
import torch.fx

class SampleNetWithViolations(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        a = x.cpu()
        b = torch.empty(x.shape)
        c = a + b + self.param
        return c

class CleanNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return x + self.param

def generate_test_cases():
    return [
        (torch.fx.symbolic_trace(SampleNetWithViolations()), 2),
        (torch.fx.symbolic_trace(CleanNet()), 0)
    ]
