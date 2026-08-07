def lora_delta_forward(x: list[list[float]], base: list[list[float]], A: list[list[float]], B: list[list[float]], scale: float) -> list[list[float]]:
    """
    x: (n, d), base: (n, d) frozen base layer output for x, A: (d, r),
    B: (r, d), scale: float.

    Returns base + scale * (x @ A) @ B as a float64 (n, d) array.
    """
    raise NotImplementedError('your code here')
