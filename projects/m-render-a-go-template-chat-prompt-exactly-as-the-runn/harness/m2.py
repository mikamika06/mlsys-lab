import ref


def check(workdir):
    from promptfmt.tokens import find_mismatched_token
    from promptfmt.gguf import recover_chat_template, compare_with_ollama

    tokens_ok = 1
    for mf, card, expected in ref.TOKEN_CASES:
        got = find_mismatched_token(mf, card)
        if got != expected:
            tokens_ok = 0
            break

    template_ok = 1
    for meta, show_out in ref.GGUF_CASES:
        recovered = recover_chat_template(meta)
        if not compare_with_ollama(recovered, show_out):
            template_ok = 0
            break

    return {
        "tokens_matched": float(tokens_ok),
        "template_match": float(template_ok)
    }
