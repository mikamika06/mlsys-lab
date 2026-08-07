def qlora_forward(x: list[list[float]], nf4_codes: list[list[int]], absmax: list[list[float]], blocksize: int, A: list[list[float]], B: list[list[float]], alpha: float) -> list[list[float]]:
    """
    QLoRA forward pass: dequantize the NF4 base weight (blockwise absmax),
    add the scaled LoRA delta B@A, then apply the resulting linear layer.
    See task.md for the exact three-step formula.
    """
    raise NotImplementedError('your code here')
