def _oracle(initial_running, pending, C):
    """Dynamic SplitFuse packing, per iteration:
      1. Seat exactly one decode token for every currently running sequence.
      2. Spend whatever token budget is left greedily on the FIFO pending
         prefill queue, chunking a request across iterations if it doesn't
         fully fit, and moving on to the next queued request (within the
         SAME iteration) if one finishes with budget still left over.
    A request that finishes prefilling this iteration joins the running
    set starting next iteration."""
    running = initial_running
    queue = list(pending)  # remaining, not-yet-prefilled length of each request, FIFO
    allocations = []

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


def _cases():
    cases = []

    # Hand-picked: exact chunk boundary that finishes one request and
    # immediately starts the next within the same iteration.
    cases.append((0, [8, 4], 8))
    # A request too large for one iteration's budget, spread over
    # several iterations while decode count stays fixed at first.
    cases.append((2, [20], 6))
    # Several running seqs already, small queue, budget barely covers decode.
    cases.append((5, [3, 3, 3], 9))
    # Running count grows mid-simulation, shrinking future prefill budget.
    cases.append((0, [4, 4, 4, 4], 5))
    # Single huge request against a tiny budget.
    cases.append((1, [37], 4))

    # Deterministic pseudo-random cases (no external RNG needed -- pure
    # integer scheduling logic).
    seeds = [1, 2, 3, 4]
    for s in seeds:
        x = s * 2654435761 % (2**32)

        def nxt(lo, hi):
            nonlocal x
            x = (x * 1103515245 + 12345) % (2**31)
            return lo + x % (hi - lo + 1)

        initial_running = nxt(0, 3)
        n = nxt(1, 5)
        pending = [nxt(1, 15) for _ in range(n)]
        C = initial_running + n + nxt(2, 10)  # guarantee running never exceeds C
        cases.append((initial_running, pending, C))

    return cases


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for initial_running, pending, C in _cases():
        total += 1
        ref = _oracle(initial_running, pending, C)
        try:
            got = sol.dynamic_splitfuse_pack(initial_running, list(pending), C)
            got = [tuple(x) for x in got]
        except Exception:
            continue
        if got == ref:
            correct += 1

    return {"exact_match": correct / total if total else 0.0}
