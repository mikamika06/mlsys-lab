import torch

def inspect_autocast_ops(model, dummy_input):
    fp16_ops = []
    fp32_ops = []

    def hook(module, input, output):
        if hasattr(output, 'dtype'):
            if output.dtype == torch.float16:
                fp16_ops.append(type(module).__name__)
            else:
                fp32_ops.append(type(module).__name__)

    hooks = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    try:
        with torch.cuda.amp.autocast():
            _ = model(dummy_input)
    finally:
        for h in hooks:
            h.remove()

    return {"fp16": list(set(fp16_ops)), "fp32": list(set(fp32_ops))}
