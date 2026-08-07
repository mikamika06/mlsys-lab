def find_mismatched_token(modelfile_tokens, modelcard_tokens):
    mf_set = set(modelfile_tokens)
    mc_set = set(modelcard_tokens)
    diff = mf_set.symmetric_difference(mc_set)
    if not diff:
        return None
    return sorted(list(diff))[0]
