import numpy as np


def constrained_free_argmax_divergence(logits, trace, allowed) -> int:
    """
    logits: (T, vocab_size) array of per-step logits.
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
    logits = np.asarray(logits, dtype=np.float64)
    vocab_size = logits.shape[1]

    count = 0
    for t, state in enumerate(trace):
        row = logits[t]
        free = int(np.argmax(row))

        allowed_tokens = sorted(set(int(tok) for tok in allowed[state]))
        mask = np.full(vocab_size, -np.inf, dtype=np.float64)
        mask[allowed_tokens] = row[allowed_tokens]
        constrained = int(np.argmax(mask))

        if free != constrained:
            count += 1

    return count
