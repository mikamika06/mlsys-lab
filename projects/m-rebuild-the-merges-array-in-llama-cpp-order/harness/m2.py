import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from llama_cpp_tok.vocab import classify_vocab_type, find_wrong_token_type

    out = {"vocab_type_match": 0.0, "wrong_token_found": 0.0}

    vocab_ok = True
    for art, expected in ref.VOCAB_TESTS:
        if classify_vocab_type(art) != expected:
            vocab_ok = False
    if vocab_ok:
        out["vocab_type_match"] = 1.0

    token_ok = True
    for tokens, expected_id in ref.TOKEN_TESTS:
        if find_wrong_token_type(tokens) != expected_id:
            token_ok = False
    if token_ok:
        out["wrong_token_found"] = 1.0

    return out
