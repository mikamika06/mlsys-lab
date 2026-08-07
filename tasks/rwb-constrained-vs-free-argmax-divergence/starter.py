def constrained_free_argmax_divergence(logits: list[list[float]], trace: list[int], allowed: dict[int, list[int]]) -> int:
    """
    logits: (T, vocab_size) array of per-step logits.
    trace: length-T list of FSM state ids active at each step.
    allowed: dict mapping FSM state id -> list of token ids allowed in
        that state.

    For each step t, compute:
      - free_argmax: argmax over the full vocab of logits[t]
      - constrained_argmax: argmax over logits[t] restricted to
        allowed[trace[t]] (ties broken by the lowest token id)

    Return the number of steps where free_argmax != constrained_argmax.
    """
    raise NotImplementedError('your code here')
