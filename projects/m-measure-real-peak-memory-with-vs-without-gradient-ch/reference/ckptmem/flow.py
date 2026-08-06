import torch
import torch.nn as nn
from typing import Dict, Any


def enable_input_require_grads(model: nn.Module):
    def make_input_require_grad(module, args, kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args[0].requires_grad_(True)
            return args, kwargs
        return args, kwargs

    target_module = None
    if hasattr(model, "layers") and len(model.layers) > 0:
        target_module = model.layers[0]
    elif len(list(model.children())) > 0:
        target_module = list(model.children())[0]

    if target_module is not None:
        target_module.register_forward_pre_hook(make_input_require_grad, with_kwargs=True)


def run_gradient_flow_experiment(
    model: nn.Module,
    x: torch.Tensor,
    freeze_base: bool,
    use_checkpointing: bool,
    use_input_require_grads: bool
) -> Dict[str, Any]:
    if freeze_base:
        if hasattr(model, "layers"):
            for p in model.layers.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = False

    if use_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    if use_input_require_grads:
        enable_input_require_grads(model)

    model.train()
    model.zero_grad(set_to_none=True)

    success = True
    error_msg = ""
    adapter_grad_norm = 0.0

    try:
        out = model(x)
        loss = out.sum()
        loss.backward()

        adapter_param = None
        if hasattr(model, "adapter") and hasattr(model.adapter, "weight"):
            adapter_param = model.adapter.weight

        if adapter_param is not None:
            if adapter_param.grad is None or float(adapter_param.grad.abs().sum().item()) == 0.0:
                success = False
                error_msg = "Adapter gradient is missing or zero"
            else:
                adapter_grad_norm = float(adapter_param.grad.norm().item())
        else:
            success = False
            error_msg = "Adapter module not found"

    except Exception as e:
        success = False
        error_msg = str(e)

    return {
        "success": success,
        "error": error_msg,
        "adapter_grad_norm": adapter_grad_norm
    }
