def total_token_count(corpus, merges):
    """
    Correct implementation of BPE token counting.
    """
    # Initialise tokens as individual characters (including whitespace)
    all_tokens = [list(s) for s in corpus]

    # Apply each merge pair in order
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

    # Return the total number of tokens across all strings
    return sum(len(t) for t in all_tokens)
