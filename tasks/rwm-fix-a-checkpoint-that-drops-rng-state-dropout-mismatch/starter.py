import random

def checkpointed_layer(x: list[list[float]], W1: list[list[float]], W2: list[list[float]], p: float, seed: int, n_pre: int, n_post: int, dY: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """Activation-checkpointed dropout block: forward once, discard the
    intermediates, then RECOMPUTE for backward under the identical dropout
    mask by saving/restoring the shared RNG's state around the block.

    Returns (Y, dX, dW1, dW2).

    BUG: this version never snapshots the RNG state at checkpoint entry, so
    the recompute below draws its mask from wherever the shared stream
    happens to be sitting after `n_post` unrelated draws -- a different mask
    than the one the original forward pass used. Fix it.
    """
    raise NotImplementedError('your code here')
