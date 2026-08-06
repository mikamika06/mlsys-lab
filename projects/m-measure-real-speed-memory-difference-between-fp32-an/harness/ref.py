import torch


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)


def get_fixture():
    torch.manual_seed(42)
    model = SyntheticModel()
    x = torch.randn(8, 32)
    return model, x
