def surviving_blocks(
    new_prompt: list[int],
    cached_seqs: list[list[int]],
    block_contents: dict[int, list[int]]
) -> list[int]:
    """
    For each sequence in cached_seqs, determine how many blocks can be fully reused.
    Returns the list of fully matched block IDs from the sequence that provides
    the maximum number of reused blocks. If there is a tie, return the one from
    the sequence that appears earliest in cached_seqs.
    """
    raise NotImplementedError
