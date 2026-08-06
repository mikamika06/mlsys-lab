import torch

def inspect_autocast_dtypes(model, x, dtype=torch.bfloat16):
    dtypes = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    dtypes.append(o.dtype)
        elif isinstance(output, torch.Tensor):
            dtypes.append(output.dtype)
    handles = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    device_type = 'cpu'
    if x.is_cuda:
        device_type = 'cuda'
    elif x.is_mps:
        device_type = 'mps'
    with torch.autocast(device_type=device_type, dtype=dtype):
        model(x)
    for h in handles:
        h.remove()
    return dtypes

def verify_weights_unchanged(model, x, dtype=torch.bfloat16):
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    device_type = 'cpu'
    if x.is_cuda:
        device_type = 'cuda'
    elif x.is_mps:
        device_type = 'mps'
    with torch.autocast(device_type=device_type, dtype=dtype):
        model(x)
    for name, p in model.named_parameters():
        if not torch.equal(initial_params[name], p):
            return False
    return True

def check_overflow(layer, x):
    device_type = 'cpu'
    if x.is_cuda:
        device_type = 'cuda'
    elif x.is_mps:
        device_type = 'mps'
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        out_fp16 = layer(x.to(torch.float16))
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out_bf16 = layer(x.to(torch.bfloat16))
    has_overflow = torch.isinf(out_fp16).any() or torch.isnan(out_fp16).any()
    safe_bf16 = not (torch.isinf(out_bf16).any() or torch.isnan(out_bf16).any())
    return has_overflow and safe_bf16
