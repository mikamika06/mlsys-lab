from prompt_lookup.draft import draft_ngram


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
