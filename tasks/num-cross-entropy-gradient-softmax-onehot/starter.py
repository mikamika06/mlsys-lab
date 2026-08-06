import math

def cross_entropy_backward(logits: list[list[float]], labels: list[int]) -> list[list[float]]:
    """Gradient of the mean softmax cross-entropy loss with respect to ``logits``.

    Args:
        logits: float64 array of shape ``(N, C)`` — raw, unnormalised scores.
        labels: int64 array of shape ``(N,)`` with values in ``[0, C)``.

    Returns:
        float64 array of shape ``(N, C)`` holding ``dL/dlogits`` for the loss
        ``L = -(1/N) * sum_i log softmax(logits_i)[labels_i]``.

    The input must not be modified in place.
    """
    raise NotImplementedError('your code here')
