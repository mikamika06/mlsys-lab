import ref


def check(workdir):
    from compiletracer.classifier import classify_snippets

    out = {"snippets_classified": 0.0, "classifications_exact": 0.0}
    want = ref.classify_snippets(ref.SNIPPETS)
    try:
        got = classify_snippets(ref.SNIPPETS)
    except Exception as e:
        out["_note"] = f"classify_snippets raised {type(e).__name__}: {e}"
        return out

    if isinstance(got, list):
        out["snippets_classified"] = float(len(got))

    if got == want:
        out["classifications_exact"] = 1.0
    else:
        out["_note"] = f"expected {want[:2]}, got {got[:2] if isinstance(got, list) else got}"

    return out
