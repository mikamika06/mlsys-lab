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
