import math
import numpy as np


def expert_choice_routing(logits: np.ndarray, capacity_factor: float):
    T, E = logits.shape
    capacity = int(math.ceil(capacity_factor * T / E))

    expert_tokens = []
    for e in range(E):
        order = np.lexsort((np.arange(T), -logits[:, e]))
        expert_tokens.append([int(x) for x in order[:capacity]])

    selected = set()
    for tokens in expert_tokens:
        selected.update(tokens)
    expert_dropped = T - len(selected)

    assignments = np.argmax(logits, axis=1)
    token_choice_dropped = 0
    for e in range(E):
        tokens = np.where(assignments == e)[0]
        order = np.lexsort((tokens, -logits[tokens, e]))
        kept = tokens[order[:capacity]]
        token_choice_dropped += len(tokens) - len(kept)

    return expert_tokens, int(expert_dropped), int(token_choice_dropped)
