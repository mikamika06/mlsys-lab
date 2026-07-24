def classify_backend(dtype, head_dim, has_attn_mask, is_causal, device_is_cpu):
    """
    Return the SDPA backend that PyTorch would select for the given arguments.
    
    Parameters
    ----------
    dtype : str
        Data type of the query/key/value tensors. Expected values are
        ``"float16"``, ``"bfloat16"``, or any other string representing a
        higher‑precision type such as ``"float32"``.
    head_dim : int
        The dimensionality of each attention head.
    has_attn_mask : bool
        Whether an arbitrary attention mask is supplied.
    is_causal : bool
        Whether the attention should be causal. This flag is ignored by the
        current selection logic but is part of the public API.
    device_is_cpu : bool
        ``True`` if the tensors are on a CPU; otherwise they are assumed to be
        on a CUDA GPU.

    Returns
    -------
    str
        One of ``"flash"``, ``"mem_efficient"``, or ``"math"``.
    """
    # Rule 1: CPU always falls back to math
    if device_is_cpu:
        return "math"
    # Rule 2: Any arbitrary mask forces the generic implementation
    if has_attn_mask:
        return "math"
    # Rule 3: Precision and head dimension determine the specialized backend
    if dtype in ("float16", "bfloat16"):
        if head_dim <= 128:
            return "flash"
        elif head_dim <= 256:
            return "mem_efficient"
    # Default fallback
    return "math"
