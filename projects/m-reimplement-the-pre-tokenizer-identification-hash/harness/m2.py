from pretokenize.disagreement import measure_disagreement

def check(workdir):
    out = {"disagreement_measured": 0.0}
    hf_tokens = [1, 2, 3, 4, 5]
    gguf_tokens = [1, 2, 9, 4, 5]
    score = measure_disagreement(hf_tokens, gguf_tokens)
    if isinstance(score, float) and 0.0 < score <= 1.0:
        out["disagreement_measured"] = 1.0
    return out
