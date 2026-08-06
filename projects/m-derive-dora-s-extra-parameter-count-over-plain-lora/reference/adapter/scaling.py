import numpy as np


def plain_lora_scaling(r: int, alpha: float) -> float:
    """Compute plain LoRA scaling factor."""
    return alpha / r


def rslora_scaling(r: int, alpha: float) -> float:
    """Compute rank-stabilized LoRA scaling factor."""
    return alpha / np.sqrt(r)
