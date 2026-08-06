def compute_scaling_factor(alpha, r, mode="lora"):
    """Compute LoRA adapter scaling factor according to lora or naive mode."""
    raise NotImplementedError


def apply_lora_scaling(x, weight_a, weight_b, alpha, r, mode="lora"):
    """Compute the LoRA delta output (x @ A.T @ B.T) multiplied by scaling factor."""
    raise NotImplementedError
