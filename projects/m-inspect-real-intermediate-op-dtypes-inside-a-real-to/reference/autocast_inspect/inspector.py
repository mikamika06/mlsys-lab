import torch

def inspect_autocast(model, x, device_type, autocast_dtype):
    activation_dtypes = []
    handles = []

    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            activation_dtypes.append(output.dtype)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook))

    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        out = model(x)

    for h in handles:
        h.remove()

    weight_dtypes = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and hasattr(module, 'weight') and module.weight is not None:
            weight_dtypes.append(module.weight.dtype)

    return {
        "output_dtype": out.dtype if isinstance(out, torch.Tensor) else None,
        "activation_dtypes": activation_dtypes,
        "weight_dtypes": weight_dtypes
    }

def synthesize_overflow():
    a = torch.full((10000,), 3.0, dtype=torch.float32)
    b = torch.full((10000,), 3.0, dtype=torch.float32)
    return a, b
