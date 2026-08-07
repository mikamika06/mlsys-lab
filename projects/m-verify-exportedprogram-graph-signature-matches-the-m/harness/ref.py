import torch
import torch.export


class StandardModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))
        self.register_buffer("running_mean", torch.zeros(8))

    def forward(self, x):
        return x @ self.weight + self.running_mean


class ContainerMutatingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer("counter", torch.zeros(1))
        self.state = [1, 2, 3]

    def forward(self, x):
        self.state.append(4)
        self.counter.add_(1.0)
        return x @ self.weight


def get_standard_test_case():
    mod = StandardModule()
    args = (torch.randn(2, 8),)
    ep = torch.export.export(mod, args)
    return mod, ep, args


def get_mutating_test_case():
    mod = ContainerMutatingModule()
    args = (torch.randn(2, 4),)
    return mod, args
