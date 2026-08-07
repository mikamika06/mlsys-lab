def constrained_free_argmax_divergence(
    logits: list[list[float]],
    trace: list[int],
    allowed: dict[int, list[int]],
) -> int:
    """logits: (T, vocab_size) list of lists of per-step logits.
    trace: length-T list of FSM state ids active at each step.
    allowed: dict mapping FSM state id -> list of token ids allowed in
        that state.

    For each step t, compute:
      - free_argmax: argmax over the full vocab of logits[t]
      - constrained_argmax: argmax over logits[t] with every token NOT in
        allowed[trace[t]] masked to -inf (the actual technique a
        grammar/FSM-constrained decoder uses), ties broken by the lowest
        token id

    Return the number of steps where free_argmax != constrained_argmax.
    """
    vocab_size = len(logits[0])

    count = 0
    for t, state in enumerate(trace):
        row = logits[t]

        free = 0
        max_val = float("-inf")
        for i in range(vocab_size):
            val = row[i]
            if val > max_val:
                max_val = val
                free = i

        allowed_set = set(int(tok) for tok in allowed[state])
        constrained = 0
        max_val_c = float("-inf")
        for i in range(vocab_size):
            if i in allowed_set:
                val = row[i]
                if val > max_val_c:
                    max_val_c = val
                    constrained = i

        if free != constrained:
            count += 1

    return count
