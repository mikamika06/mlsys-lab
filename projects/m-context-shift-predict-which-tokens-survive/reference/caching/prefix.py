def longest_surviving_prefix(cached_prompts: list[list[int]], new_prompt: list[int]) -> int:
    best = 0
    for cp in cached_prompts:
        match_len = 0
        for ct, nt in zip(cp, new_prompt):
            if ct == nt:
                match_len += 1
            else:
                break
        if match_len > best:
            best = match_len
    return best
