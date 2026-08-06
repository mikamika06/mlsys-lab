import torch


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)


class MutationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.val = 0

    def forward(self, x):
        self.val += 1
        return x + self.val


def get_test_exported_program():
    model = DummyModel()
    x = torch.randn(4, 4)
    return torch.export.export(model, (x,))


def get_mutation_func():
    model = MutationModel()
    x = torch.randn(2, 2)
    return model, (x,)
