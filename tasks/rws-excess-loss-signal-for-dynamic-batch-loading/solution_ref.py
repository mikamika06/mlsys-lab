def excess_loss_signal(current_losses: list[float],
                       reference_losses: list[float]) -> list[float]:
    """
    Compute the element‑wise difference between current and reference losses.
    The result is a list of floats of the same length as the inputs.
    """
    return [c - r for c, r in zip(current_losses, reference_losses)]
