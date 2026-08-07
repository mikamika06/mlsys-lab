import torch


def measure_8bit_delta(model, optimizer_cls_32, optimizer_cls_8):
    """
    Instantiate both optimizers on the model's trainable parameters.
    Simulate a step by assigning dummy gradients and calling step().

    Returns: {"bytes_32": int, "bytes_8": int, "delta_bytes": int}
    """
    raise NotImplementedError


def verify_qlora_optimizer_clean(model, optimizer):
    """
    Returns False if any parameter where requires_grad=False has
    a non-empty state dictionary in optimizer.state. Returns True otherwise.
    """
    raise NotImplementedError
