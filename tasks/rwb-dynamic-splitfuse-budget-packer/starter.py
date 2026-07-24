def dynamic_splitfuse_pack(initial_running: int, pending: list[int], C: int) -> list[tuple[int, int]]:
    """
    Simulate DeepSpeed-FastGen's Dynamic SplitFuse scheduling.

    initial_running: number of decode sequences already running before
        the first iteration.
    pending: FIFO queue of prefill request prompt lengths, not yet
        started.
    C: max_num_batched_tokens, the total token budget per iteration.

    Runs until the pending queue is fully drained. Returns the list of
    (decode_tokens, prefill_chunk_tokens) pairs, one per iteration.
    """
    raise NotImplementedError('your code here')
