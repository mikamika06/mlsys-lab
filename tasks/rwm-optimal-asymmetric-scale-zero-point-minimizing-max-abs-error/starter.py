def derive_affine_qparams(x: list[float], nbits: int) -> tuple[float, int]:
    """
    Return (scale: float, zero_point: int) for asymmetric min-max affine
    quantization of `x` to `nbits` unsigned bits, as described in task.md.
    """
    raise NotImplementedError('your code here')
