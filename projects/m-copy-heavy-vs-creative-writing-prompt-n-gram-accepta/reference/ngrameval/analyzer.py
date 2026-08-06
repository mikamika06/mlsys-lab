def extract_ngram_matches(prompt_tokens, target_tokens, n=4):
    if len(target_tokens) < n:
        return []
    prompt_ngrams = set()
    for i in range(len(prompt_tokens) - n + 1):
        prompt_ngrams.add(tuple(prompt_tokens[i:i+n]))

    matches = []
    for i in range(len(target_tokens) - n + 1):
        gram = tuple(target_tokens[i:i+n])
        if gram in prompt_ngrams:
            matches.append((i, n))
    return matches
