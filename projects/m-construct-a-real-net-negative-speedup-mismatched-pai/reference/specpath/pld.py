def detect_ngram_loop(tokens, max_n=4):
    n_total = len(tokens)
    for n in range(1, max_n + 1):
        if n_total >= 2 * n:
            pattern = tokens[-n:]
            prev = tokens[-2 * n:-n]
            if pattern == prev:
                return True, n
    return False, 0


def generate_pld_draft(prompt_tokens, max_k, n_gram_len):
    draft = []
    curr = list(prompt_tokens)

    for _ in range(max_k):
        if len(curr) < n_gram_len:
            break

        match_ngram = curr[-n_gram_len:]
        found_next_token = None

        search_limit = len(curr) - n_gram_len
        for i in range(search_limit):
            if curr[i:i + n_gram_len] == match_ngram:
                found_next_token = curr[i + n_gram_len]
                break

        if found_next_token is None:
            break

        draft.append(found_next_token)
        curr.append(found_next_token)

        is_loop, _ = detect_ngram_loop(curr, max_n=min(4, len(curr) // 2))
        if is_loop:
            break

    return draft
