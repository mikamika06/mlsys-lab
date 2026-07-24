def pick_backend(head_dim, dtype, mask_type, causal):
    """
    Return the eligible SDPA backend for the given constraints.
    
    Parameters
    ----------
    head_dim : int
        Dimensionality of each attention head.
    dtype : str
        Data type of the tensors. Expected values are "float16", "bfloat16" or "float32".
    mask_type : str
        Either "causal" or "full".
    causal : bool
        Whether a causal mask should be applied.

    Returns
    -------
    str
        One of "flash", "mem_efficient" or "math".
    """
    if head_dim % 64 == 0 and dtype in {"float16", "bfloat16"} and mask_type == "causal" and causal:
        return "flash"
    elif head_dim <= 256 and dtype == "float32" and mask_type == "full":
        return "mem_efficient"
    else:
        return "math"
