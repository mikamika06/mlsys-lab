import torch


def check_activation_retention(model, inputs):
    with torch.no_grad():
        out_no_grad = model(inputs)
    has_grad_no_grad = any(tensor.requires_grad for tensor in out_no_grad if isinstance(tensor, torch.Tensor))

    out_with_grad = model(inputs)
    has_grad_with_grad = any(tensor.requires_grad for tensor in out_with_grad if isinstance(tensor, torch.Tensor))

    return {
        "retained_without_nograd": float(has_grad_with_grad),
        "cleared_with_nograd": float(not has_grad_no_grad)
    }
