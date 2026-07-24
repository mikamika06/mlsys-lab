def _bpe_token_count(corpus, merges):
    """
    Reference implementation of BPE token counting.
    """
    # Start with character tokens (including whitespace)
    all_tokens = []
    for s in corpus:
        all_tokens.append(list(s))

    # Apply merges sequentially
    for a, b in merges:
        new_all = []
        for tokens in all_tokens:
            i = 0
            merged = []
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            new_all.append(merged)
        all_tokens = new_all

    # Count total tokens
    return sum(len(t) for t in all_tokens)


def grade(sol, fx) -> dict:
    """
    Grade the candidate solution by comparing its output to a reference
    implementation on several test cases.
    """
    tests = [
        (["ab", "ba"], [("a", "b")], 2),
        (["abc"], [("a", "b"), ("ab", "c")], 1),
        (["ab c", "c ab"], [("a", "b")], 6),  # tokens: ['ab',' ','c'] and ['c',' ','ab']
    ]

    ok = 1.0
    for corpus, merges, expected in tests:
        try:
            got = sol.total_token_count(list(corpus), list(merges))
        except Exception:
            return {"exact_match": 0.0}
        ref = _bpe_token_count(corpus, merges)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
