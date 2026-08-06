import math

def masked_softmax(logits: list[list[float]], mask: list[list[int]]) -> list[list[float]]:
    """WRONG IMPLEMENTATION: zeroes logits where mask==0 before softmax.
This produces a distribution that does not match the correct additive -inf masking."""
    raise NotImplementedError('your code here')
