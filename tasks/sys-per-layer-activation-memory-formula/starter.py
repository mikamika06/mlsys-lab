def activation_memory_bytes(b, s, h, a):
    """Per-transformer-layer activation memory in bytes, no recomputation.

    b: micro-batch size. s: sequence length. h: hidden size.
    a: number of attention heads.
    Returns s*b*h*(34 + 5*a*s/h) as a float.
    """
    raise NotImplementedError('your code here')
