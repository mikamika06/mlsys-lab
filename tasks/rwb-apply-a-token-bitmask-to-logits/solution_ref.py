import math
from collections.abc import Iterable

def masked_greedy(logits: list[list[float]],
                  allowed_sets: Iterable[Iterable[int]]) -> list[int]:
    """
    Return the greedy token indices after masking disallowed logits.

    Parameters
    ----------
    logits : list of list of float
        Logit scores for each token at each decoding step.
    allowed_sets : Iterable[Iterable[int]]
        For each step a collection of token indices that are permitted.

    Returns
    -------
    list of int
        The index of the chosen token for each step.
    """
    result = []
    for i, allowed in enumerate(allowed_sets):
        allowed_set = set(allowed)
        best_val = -math.inf
        best_idx = 0
        first = True
        for j, val in enumerate(logits[i]):
            if j in allowed_set:
                if first or val > best_val:
                    best_val = val
                    best_idx = j
                    first = False
        result.append(best_idx)
    return result
