import math
import numpy as np


def expert_choice_routing(logits: np.ndarray, capacity_factor: float):
    T, E = logits.shape
    capacity = int(math.ceil(capacity_factor * T / E))

    expert_tokens = []
    for e in range(E):
        order_t = sorted(range(T), key=lambda t: (-logits[t, e], t))
        expert_tokens.append([int(x) for x in order_t[:capacity]])

    selected = set()
    for tokens in expert_tokens:
        selected.update(tokens)
    expert_dropped = T - len(selected)

    assignments = []
    for t in range(T):
        best_e = 0
        best_val = logits[t, 0]
        for e in range(1, E):
            val = logits[t, e]
            if val > best_val:
                best_val = val
                best_e = e
        assignments.append(best_e)

    token_choice_dropped = 0
    for e in range(E):
        tokens = [t for t in range(T) if assignments[t] == e]
        order_tokens = sorted(tokens, key=lambda t: (-logits[t, e], t))
        kept = order_tokens[:capacity]
        token_choice_dropped += len(tokens) - len(kept)

    return expert_tokens, int(expert_dropped), int(token_choice_dropped)
