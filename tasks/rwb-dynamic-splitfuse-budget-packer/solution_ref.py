def dynamic_splitfuse_pack(initial_running: int, pending: list[int], C: int) -> list[tuple[int, int]]:
    """
    Simulate DeepSpeed-FastGen's Dynamic SplitFuse scheduling.

    initial_running: number of decode sequences already running before
        the first iteration.
    pending: FIFO queue of prefill request prompt lengths, not yet
        started.
    C: max_num_batched_tokens, the total token budget per iteration.

    Each iteration:
      1. Seat exactly one decode token per currently running sequence
         (unconditionally -- decode always gets priority).
      2. Spend the remaining budget greedily on the pending prefill
         queue in FIFO order: chunk a request across iterations if it
         doesn't fully fit in the remaining budget, and move on to the
         next queued request (within the SAME iteration) if one
         finishes with budget still left over.
      A request that finishes prefilling in an iteration joins the
      running set starting the NEXT iteration.

    Runs until the pending queue is fully drained. Returns the list of
    (decode_tokens, prefill_chunk_tokens) pairs, one per iteration.
    """
    running = initial_running
    queue = list(pending)
    allocations: list[tuple[int, int]] = []

    while queue:
        decode_tokens = running
        budget = C - decode_tokens
        prefill_chunk_tokens = 0
        newly_completed = 0

        while budget > 0 and queue:
            take = min(budget, queue[0])
            prefill_chunk_tokens += take
            budget -= take
            queue[0] -= take
            if queue[0] == 0:
                queue.pop(0)
                newly_completed += 1

        allocations.append((decode_tokens, prefill_chunk_tokens))
        running += newly_completed

    return allocations
