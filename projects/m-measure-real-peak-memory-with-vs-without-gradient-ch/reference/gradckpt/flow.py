import torch
import torch.utils.checkpoint

def check_gradient_flow_frozen(model, input_tensor):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    x_in = input_tensor.detach().clone().requires_grad_(True)
    if hasattr(model, "layers"):
        out = x_in
        for layer in model.layers:
            out = torch.utils.checkpoint.checkpoint(layer, out, use_reentrant=False)
        out = model.head(out)
    else:
        out = model(x_in)
    out.sum().backward()

    grads = [p.grad for p in model.parameters()]
    return all(g is None for g in grads)
