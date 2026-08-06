import torch
from scalerlab.clip import perform_optimizer_step


def test_clip_before_unscale_bug():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    scale = 10.0
    max_norm = 1.0
    model[0].weight.grad = torch.tensor([[100.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    class MockScaler:
        def __init__(self, s):
            self.s = s
        def unscale_(self, opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.div_(self.s)
        def step(self, opt):
            opt.step()
        def update(self):
            pass

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = MockScaler(scale)

    perform_optimizer_step(model, optimizer, scaler, max_norm)

    final_grad = model[0].weight.grad
    expected_grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(final_grad, expected_grad, atol=1e-4)


def test_underflow_counter_behavior():
    from scalerlab.counter import UnderflowTracker
    tracker = UnderflowTracker()
    tracker.update(True)
    assert tracker.get_skipped_count() == 1
