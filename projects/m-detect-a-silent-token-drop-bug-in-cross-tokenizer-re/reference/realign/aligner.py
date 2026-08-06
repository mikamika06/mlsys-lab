def align_tokens(draft_tokens, target_vocab_map):
    aligned = []
    for t in draft_tokens:
        mapped = target_vocab_map.get(t)
        if mapped is not None:
            aligned.append(mapped)
    return aligned
