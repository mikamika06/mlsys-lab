def compare_chunked_vs_monolithic(prompt_lens: list[int], C: int) -> dict:
    """Simulate two prefill scheduling policies over N requests, processed
    strictly in FIFO (arrival) order, under a fixed per-iteration token
    budget C. Both policies batch as many requests together per iteration
    as the budget allows, in order; they differ in whether a single
    request's prompt may be SPLIT across iteration boundaries.

    Monolithic (no chunking): a request's whole prompt must be processed
    in one iteration. Starting from the current budget C, keep adding the
    next pending request's FULL prompt while it still fits; stop (defer
    the rest to the next iteration) at the first one that doesn't. A
    request whose prompt exceeds C entirely still runs, alone, in its own
    iteration (chunking is not available to save it).

    Chunked: a request's prompt may be split across iterations. Each
    iteration greedily consumes up to C tokens from the front of the
    FIFO queue, taking as many tokens as fit from the current head
    request, finishing it and moving to the next if there's budget left,
    or leaving it partially done (to resume first next iteration) if the
    budget runs out mid-request. This never wastes budget.

    prompt_lens : list of N positive ints (prompt length per request, in
                  FIFO arrival order).
    C           : positive int, the per-iteration token budget.

    Returns a dict:
      "iters_mono"    : int, total iterations for monolithic scheduling.
      "iters_chunked" : int, total iterations for chunked scheduling.
      "ttft_mono"     : list[int] length N, 1-indexed iteration at which
                        request i's prefill completes (monolithic).
      "ttft_chunked"  : list[int] length N, 1-indexed iteration at which
                        request i's prefill completes (chunked).
    """
    raise NotImplementedError('your code here')
