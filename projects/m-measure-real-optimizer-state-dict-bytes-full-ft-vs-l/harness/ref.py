import torch


class DummyModel(torch.nn.Module):
    def __init__(self, dim, with_lora=False, rank=4):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        if with_lora:
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
            self.lora_A = torch.nn.Parameter(torch.randn(dim, rank))
            self.lora_B = torch.nn.Parameter(torch.randn(rank, dim))

    def forward(self, x):
        return self.linear(x)


class AdamW8bitMock(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, dict(lr=lr))

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.0)
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.int8)
                        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.int8)


def step_and_count(model, opt_cls):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = opt_cls(trainable)
    for p in trainable:
        p.grad = torch.ones_like(p)
    opt.step()

    total = 0
    for state_dict in opt.state.values():
        for v in state_dict.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total
