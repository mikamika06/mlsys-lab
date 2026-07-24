import numpy as np


def propose_tokens(
    context: np.ndarray,
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    num_speculative_tokens: int,
) -> np.ndarray:
    """Longest-suffix-match (n-gram / prompt-lookup) speculative proposer.

    Search n-gram length L from prompt_lookup_max down to
    prompt_lookup_min for the most recent non-overlapping earlier
    occurrence of the current suffix, and propose the tokens that followed
    it (empty array if no length matches)."""
    raise NotImplementedError('your code here')
