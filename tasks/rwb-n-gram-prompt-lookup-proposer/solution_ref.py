import numpy as np


def propose_tokens(
    context: np.ndarray,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    num_speculative_tokens: int,
) -> np.ndarray:
    """Longest-suffix-match (n-gram / prompt-lookup) speculative proposer.

    Tries n-gram length L from prompt_lookup_max down to prompt_lookup_min;
    for the first L with a non-overlapping earlier occurrence of the
    current suffix, proposes the tokens that followed the most recent such
    occurrence. Empty proposal if no length matches.
    """
    context = np.asarray(context)
    n = len(context)

    for L in range(prompt_lookup_max, prompt_lookup_min - 1, -1):
        if L <= 0 or L > n:
            continue
        suffix = context[n - L:n]
        best_i = -1
        max_i = n - 2 * L
        for i in range(0, max_i + 1):
            if np.array_equal(context[i:i + L], suffix):
                best_i = i
        if best_i >= 0:
            start = best_i + L
            return context[start:start + num_speculative_tokens]

    return np.array([], dtype=context.dtype)
