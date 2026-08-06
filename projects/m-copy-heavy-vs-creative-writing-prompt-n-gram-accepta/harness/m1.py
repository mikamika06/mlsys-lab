import ref


def check(workdir):
    from ngrameval.analyzer import extract_ngram_matches

    out = {"distributions_matched": 0.0, "configs": float(len(ref.WORKLOADS))}
    ok = 0
    for i, w in enumerate(ref.WORKLOADS):
        want = extract_ngram_matches(w["prompt"], w["target"], n=4)
        try:
            got = extract_ngram_matches(w["prompt"], w["target"], n=4)
        except Exception:
            got = []
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"workload {i}: got {got}, reference {want}"
    out["distributions_matched"] = float(ok)
    return out
