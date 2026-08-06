import torch


def verify_nested_disabled_autocast(model, inputs):
    model.eval()
    device = next(model.parameters()).device
    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtypes = []

    def hook(module, args, output):
        if isinstance(output, torch.Tensor):
            dtypes.append(output.dtype)
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    dtypes.append(o.dtype)

    handles = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]

    try:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
            with torch.autocast(device_type=device_type, enabled=False):
                with torch.no_grad():
                    _ = model(inputs)
    finally:
        for h in handles:
            h.remove()

    all_fp32 = all(d == torch.float32 for d in dtypes) if dtypes else True
    return {
        "all_fp32": bool(all_fp32),
        "dtypes": [str(d) for d in dtypes],
    }
