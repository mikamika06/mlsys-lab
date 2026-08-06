import torch
from capturer.step import CapturedStep


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x)


def get_reference_step_output(x, y):
    torch.manual_seed(42)
    model = DummyModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    step = CapturedStep(model, optimizer, loss_fn)
    step.capture(x, y)

    # Replay twice with new data
    x2 = x + 0.1
    y2 = y + 0.1
    out, loss = step.replay(x2, y2)
    return out, loss
