def propose_tokens(context: list[int], prompt_lookup_min: int, prompt_lookup_max: int, num_speculative_tokens: int) -> list[int]:
    """Longest-suffix-match (n-gram / prompt-lookup) speculative proposer.

    Search n-gram length L from prompt_lookup_max down to
    prompt_lookup_min for the most recent non-overlapping earlier
    occurrence of the current suffix, and propose the tokens that followed
    it (empty array if no length matches)."""
    raise NotImplementedError('your code here')
