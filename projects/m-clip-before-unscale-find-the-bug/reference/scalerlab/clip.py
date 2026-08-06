import torch


def perform_optimizer_step(model, optimizer, scaler, max_norm):
    """Performs unscaling, gradient clipping, and optimizer step in correct order."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
