def _oracle(candidates, target_probs, threshold):
    accepted = []
    best_index = 0
    best_len = -1
    best_path = []

    for idx, path in enumerate(candidates):
        prefix = []
        for pos, token in enumerate(path):
            if pos >= len(target_probs):
                break
            prob = target_probs[pos].get(token, 0.0)
            if prob < threshold:
                break
            prefix.append(token)

        if len(prefix) > 0:
            accepted.append(idx)

        if len(prefix) > best_len:
            best_len = len(prefix)
            best_index = idx
            best_path = prefix

    return best_index, best_path, accepted


def grade(sol, fx) -> dict:
    cases = [
        (
            [[4, 8, 9], [4, 7], [3, 2]],
            [{4: 0.9, 3: 0.8}, {8: 0.7, 7: 0.2, 2: 0.6}, {9: 0.4}],
            0.5,
        ),
        (
            [[1, 2], [5, 6, 7], [8]],
            [{1: 0.5, 5: 0.5, 8: 0.9}, {2: 0.4, 6: 0.8}, {7: 0.8}],
            0.5,
        ),
        (
            [[10, 11, 12], [20, 21, 22], [30]],
            [{10: 0.9, 20: 0.9, 30: 0.1}, {11: 0.9, 21: 0.9}, {12: 0.9, 22: 0.9}],
            0.2,
        ),
        (
            [[99], [1, 99], []],
            [{1: 0.6}, {99: 0.6}],
            0.5,
        ),
    ]

    ok = 1.0
    for candidates, probs, threshold in cases:
        try:
            got = sol.verify_medusa_candidates(candidates, probs, threshold)
            got = (got[0], list(got[1]), list(got[2]))
        except Exception:
            ok = 0.0
            break

        ref = _oracle(candidates, probs, threshold)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
