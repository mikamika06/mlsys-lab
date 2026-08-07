import torch

def find_first_nan_tensor(model, inputs):
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            outs = output if isinstance(output, tuple) else (output,)
            for o in outs:
                if isinstance(o, torch.Tensor) and (torch.isnan(o).any() or torch.isinf(o).any()):
                    activations['faulty'] = name
                    break
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.cuda.amp.autocast():
            _ = model(*inputs)
    finally:
        for h in hooks:
            h.remove()

    return activations.get('faulty', None)
