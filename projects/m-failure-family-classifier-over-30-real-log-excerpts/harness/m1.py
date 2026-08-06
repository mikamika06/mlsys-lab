import ref


def check(workdir):
    from faildebug.classifier import classify_excerpts
    out = {"classified_match": 0.0}
    excerpts = ref.EXCERPTS
    got = classify_excerpts(excerpts)
    want = ref.classify_excerpts(excerpts)
    if got == want:
        out["classified_match"] = 1.0
    else:
        out["_note"] = f"Expected {want[:3]}, got {got[:3]}"
    return out
