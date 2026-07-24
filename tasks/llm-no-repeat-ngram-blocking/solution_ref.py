from typing import List, Set

def no_repeat_ngram_blocking(prev_tokens: list[int], n: int) -> set[int]:
    """
    Return the set of token ids that would create a repeated n‑gram if appended to prev_tokens.
    """
    m = len(prev_tokens)
    if m < n - 1:
        return set()
    last_gram = tuple(prev_tokens[-(n-1):]) if n > 1 else ()
    banned: Set[int] = set()
    for i in range(m - n + 1):
        gram = tuple(prev_tokens[i:i+n-1])
        if gram == last_gram:
            banned.add(prev_tokens[i+n-1])
    return banned
