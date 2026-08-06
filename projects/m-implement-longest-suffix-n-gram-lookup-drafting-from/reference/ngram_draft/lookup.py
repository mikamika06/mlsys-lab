def find_longest_suffix_match(tokens, max_ngram_size, draft_len):
    if len(tokens) < 2:
        return []
    target = tokens
    max_k = min(max_ngram_size, len(target) - 1)
    for k in range(max_k, 0, -1):
        ngram = target[-k:]
        search_limit = len(target) - k
        for i in range(search_limit - 1, -1, -1):
            if target[i : i + k] == ngram:
                match_start = i + k
                match_end = min(len(target), match_start + draft_len)
                draft = target[match_start:match_end]
                if len(draft) > 0:
                    return draft
    return []
