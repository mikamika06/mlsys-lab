import torch


def profile_zerograd_allocation(model, inputs):
    """Profiles differences in memory and gradient tensor presence for zero_grad options."""
    out = model(inputs)
    loss = out.sum()
    loss.backward()

    retained_grads_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    retained_grads_count = sum(1 for p in model.parameters() if p.grad is not None)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    zero_fill_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    zero_fill_count = sum(1 for p in model.parameters() if p.grad is not None)

    for p in model.parameters():
        p.grad = None

    none_fill_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    )
    none_fill_count = sum(1 for p in model.parameters() if p.grad is not None)

    return {
        "retained_grads_bytes": retained_grads_bytes,
        "retained_grads_count": retained_grads_count,
        "zero_fill_bytes": zero_fill_bytes,
        "zero_fill_count": zero_fill_count,
        "none_fill_bytes": none_fill_bytes,
        "none_fill_count": none_fill_count,
        "allocated_bytes_saved": zero_fill_bytes - none_fill_bytes,
    }
