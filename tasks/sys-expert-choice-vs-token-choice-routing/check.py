import math
import numpy as np


def _oracle(logits, capacity_factor):
    T, E = logits.shape
    capacity = int(math.ceil(capacity_factor * T / E))

    expert_tokens = []
    for e in range(E):
        order = np.lexsort((np.arange(T), -logits[:, e]))
        expert_tokens.append([int(x) for x in order[:capacity]])

    selected = set()
    for tokens in expert_tokens:
        selected.update(tokens)
    expert_dropped = int(T - len(selected))

    assignments = np.argmax(logits, axis=1)
    token_choice_dropped = 0
    for e in range(E):
        tokens = np.where(assignments == e)[0]
        order = np.lexsort((tokens, -logits[tokens, e]))
        kept = tokens[order[:capacity]]
        token_choice_dropped += int(len(tokens) - len(kept))

    return expert_tokens, expert_dropped, token_choice_dropped


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    [3.0, 1.0],
                    [2.0, 4.0],
                    [0.5, 2.5],
                ]
            ),
            1.0,
        ),
        (
            np.array(
                [
                    [5.0, 5.0, 1.0],
                    [4.0, 2.0, 3.0],
                    [1.0, 6.0, 2.0],
                    [3.0, 1.0, 7.0],
                    [2.0, 3.0, 4.0],
                ]
            ),
            0.8,
        ),
        (
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            1.5,
        ),
    ]

    ok = 1.0
    for logits, factor in cases:
        try:
            got = sol.expert_choice_routing(logits, factor)
            got = (list(map(list, got[0])), int(got[1]), int(got[2]))
        except Exception:
            ok = 0.0
            break

        ref = _oracle(logits, factor)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
