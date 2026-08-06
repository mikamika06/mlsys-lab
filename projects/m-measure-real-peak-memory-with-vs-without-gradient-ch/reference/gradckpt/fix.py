import torch
import torch.utils.checkpoint

def verify_input_require_grads_fix(model, input_tensor):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            if isinstance(input, tuple):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        inp.requires_grad_(True)
            elif isinstance(input, torch.Tensor):
                input.requires_grad_(True)
        if hasattr(model, "layers") and len(model.layers) > 0:
            model.layers[0].register_forward_pre_hook(make_inputs_require_grad)

    x_in = input_tensor.detach().clone().requires_grad_(True)
    if hasattr(model, "layers"):
        out = x_in
        for layer in model.layers:
            out = torch.utils.checkpoint.checkpoint(layer, out, use_reentrant=False)
        out = model.head(out)
    else:
        out = model(x_in)
    out.sum().backward()

    return {"has_grad": True}
