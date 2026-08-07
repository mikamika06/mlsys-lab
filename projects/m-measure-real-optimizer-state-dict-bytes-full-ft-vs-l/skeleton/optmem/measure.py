import torch


def get_optimizer_state_bytes(optimizer):
    """
    Returns the total bytes occupied by all torch.Tensor objects
    inside the optimizer's state dictionary.
    """
    raise NotImplementedError


def compare_full_vs_lora(full_model, lora_model):
    """
    Instantiate torch.optim.AdamW for both models, capturing only parameters
    that require gradients. Simulate a step by assigning a dummy gradient
    (e.g., torch.ones_like(p)) to all trainable parameters and calling step().

    Returns: {"full_bytes": int, "lora_bytes": int, "size_ratio": float}
    """
    raise NotImplementedError
