import torch

def get_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )

def get_input():
    torch.manual_seed(42)
    return torch.randn(8, 16, dtype=torch.float32)

def check_m1_oracle(model, x):
    activation_dtypes = []
    handles = []

    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            activation_dtypes.append(output.dtype)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
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
