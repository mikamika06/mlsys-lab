import torch


def get_test_cases():
    torch.manual_seed(42)
    cases = []
    for scale in [1.0, 1024.0, 0.001]:
        param = torch.nn.Parameter(torch.tensor([10.0, 20.0], dtype=torch.float32))
        grad = torch.tensor([100.0, -200.0], dtype=torch.float32) * scale
        param.grad = grad.clone()
        cases.append({"param": param, "scale": scale, "max_norm": 1.0})
    return cases


def simulate_reference_step(param, scale, max_norm):
    p = torch.nn.Parameter(param.clone())
    p.grad = param.grad.clone()
    optimizer = torch.optim.SGD([p], lr=0.1)

    class DummyScaler:
        def __init__(self, s):
            self.s = s
        def unscale_(self, opt):
            for group in opt.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.mul_(1.0 / self.s)
        def step(self, opt):
            opt.step()
        def update(self):
            pass

    scaler = DummyScaler(scale)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p], max_norm)
    return p.grad.clone()
