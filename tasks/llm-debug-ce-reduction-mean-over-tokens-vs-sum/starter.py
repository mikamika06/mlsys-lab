import math

def cross_entropy_loss(
    logits: list[list[list[float]]],
    targets: list[list[int]],
    mask: list[list[bool]] | None = None,
) -> list[float]:
    raise NotImplementedError('your code here')
