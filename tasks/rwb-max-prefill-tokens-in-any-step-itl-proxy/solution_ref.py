def _chunked_max(prompt_lengths, decode_load, budget):
    queue = [int(p) for p in prompt_lengths]
    t = 0
    max_step = 0
    while queue:
        d = decode_load[t] if t < len(decode_load) else 0
        remaining = max(budget - d, 0)
        step_prefill = 0
        while queue and remaining > 0:
            take = min(queue[0], remaining)
            queue[0] -= take
            remaining -= take
            step_prefill += take
            if queue[0] == 0:
                queue.pop(0)
        max_step = max(max_step, step_prefill)
        t += 1
    return max_step


def _unchunked_max(prompt_lengths):
    return max((int(p) for p in prompt_lengths), default=0)


def prefill_chunking_max_step(prompt_lengths: list, decode_load: list, budget: int):
    """
    prompt_lengths: FIFO queue of pending prefill jobs' prompt token
        counts.
    decode_load: decode_load[t] = number of tokens that already-running
        decode requests unconditionally consume at iteration t (they
        can never be delayed by the scheduler; use 0 for iterations
        beyond the given list, i.e. once decode has drained).
    budget: max total tokens (decode + prefill) processed in one
        iteration under the CHUNKED prefill policy.

    Simulate a continuous-batching scheduler under two policies and
    return (max_prefill_tokens_chunked, max_prefill_tokens_unchunked):
    the maximum number of prefill tokens processed in any single
    iteration.

    - Chunked: each iteration, decode_load[t] tokens run unconditionally;
      whatever budget remains is spent taking (possibly partial) tokens
      off the front of the prefill queue, packing multiple jobs into one
      iteration if they fit.
    - Unchunked: each queued job is prefilled to completion in a single
      iteration with no cap, so the worst single iteration is exactly
      the largest prompt in the workload.
    """
    return (
        _chunked_max(prompt_lengths, decode_load, budget),
        _unchunked_max(prompt_lengths),
    )
