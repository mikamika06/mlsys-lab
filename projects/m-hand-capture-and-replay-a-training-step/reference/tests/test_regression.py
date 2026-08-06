import sys
import torch

sys.path.insert(0, ".")
from capturer.step import CapturedStep


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_graph_buffer_safety():
    if not torch.cuda.is_available():
        return
    model = SimpleModel().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    step_runner = CapturedStep(model, opt, loss_fn)
    x = torch.randn(2, 4, device="cuda")
    y = torch.randn(2, 4, device="cuda")

    step_runner.capture(x, y)
    out1, loss1 = step_runner.replay(x, y)
    out2, loss2 = step_runner.replay(x, y)

    assert not torch.allclose(out1, out2), "outputs should reflect updated weights across replays"
