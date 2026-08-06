import torch
import ref


def check(workdir):
    from scalerlab.clip import perform_optimizer_step

    cases = ref.get_test_cases()
    max_err = 0.0

    for case in cases:
        param = case["param"]
        scale = case["scale"]
        max_norm = case["max_norm"]

        model = torch.nn.Sequential()
        p_ref = torch.nn.Parameter(param.clone())
        p_ref.grad = param.grad.clone()
        model.add_module("p", torch.nn.Linear(2, 2, bias=False))
        model.p.weight = p_ref

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

        scaler = MockScaler(scale)
        want_grad = ref.simulate_reference_step(param, scale, max_norm)

        perform_optimizer_step(model, optimizer, scaler, max_norm)
        got_grad = p_ref.grad

        err = torch.max(torch.abs(got_grad - want_grad)).item()
        if err > max_err:
            max_err = err

    out = {"max_abs_err": float(max_err)}
    return out
