import ref

def check(workdir):
    from sloclassify.report import generate_report
    out = {"classification_match": 0.0}
    want = ref.generate_report(ref.REQUESTS, ref.SLO_TARGET)
    try:
        got = generate_report(ref.REQUESTS, ref.SLO_TARGET)
        if got == want:
            out["classification_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = str(e)
    return out
