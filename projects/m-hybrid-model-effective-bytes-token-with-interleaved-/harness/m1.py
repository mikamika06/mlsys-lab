import ref


def check(workdir):
    from hybridkv.config import classify_attention
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for layer in cfg["layers"]:
            total += 1
            want = ref.classify_attention(layer)
            got = classify_attention(layer)
            if want == got:
                ok += 1
    out = {"classification_matched": float(ok)}
    return out
