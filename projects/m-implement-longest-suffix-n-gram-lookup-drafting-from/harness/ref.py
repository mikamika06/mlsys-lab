def draft_ngram(tokens: list[int], max_n: int, max_draft_len: int) -> list[int]:
    if not tokens or max_n <= 0 or max_draft_len <= 0:
        return []
    for n in range(min(max_n, len(tokens)), 0, -1):
        suffix = tokens[-n:]
        search_space = len(tokens) - n
        for i in range(search_space - 1, -1, -1):
            if tokens[i:i+n] == suffix:
                draft_start = i + n
                draft = tokens[draft_start : draft_start + max_draft_len]
                if draft:
                    return draft
    return []


def simulate(prompt: list[int], target: list[int], max_n: int, max_draft_len: int) -> dict:
    current = list(prompt)
    steps = 0
    accepted_draft_tokens = 0
    while len(current) < len(target):
        steps += 1
        draft = draft_ngram(current, max_n, max_draft_len)
        remaining = len(target) - len(current)
        draft = draft[:remaining]
        match_count = 0
        for i, d in enumerate(draft):
            if d == target[len(current) + i]:
                match_count += 1
            else:
                break
        accepted_draft_tokens += match_count
        current.extend(draft[:match_count])
        if len(current) < len(target):
            current.append(target[len(current)])
    return {"steps": steps, "accepted": accepted_draft_tokens, "generated": current}


M1_INPUTS = [
    ([1, 2, 3, 4, 5, 2, 3], 2, 3),
    ([9, 9, 9, 9, 9], 3, 2),
    ([1, 2, 3], 2, 2),
    ([1, 2, 1, 2, 1, 2], 2, 4),
    ([5, 6, 7, 8, 9, 10, 6, 7, 8], 3, 2)
]

M1_CASES = [(t, n, d, draft_ngram(t, n, d)) for t, n, d in M1_INPUTS]

M2_INPUTS = [
    ([1, 2], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], 2, 2),
    ([5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 3, 4),
    ([1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8], 2, 2),
    ([1, 2, 3, 4], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7], 4, 5)
]

M2_CASES = [
    (p, t, n, d, simulate(p, t, n, d)["steps"], simulate(p, t, n, d)["accepted"])
    for p, t, n, d in M2_INPUTS
]
