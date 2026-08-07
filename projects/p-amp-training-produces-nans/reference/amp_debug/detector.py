import torch

def register_nan_detector(model):
    faulty_modules = []

    def check_tensor(t, name):
        if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
            if name not in faulty_modules:
                faulty_modules.append(name)

    def make_hook(name):
        return lambda mod, inp, out: check_tensor(out, name)

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    return model, faulty_modules, hooks
