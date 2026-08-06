import ref

def check(workdir):
    from profilediag.triage import triage_transcripts
    out = {"triage_accuracy": 0.0}
    res = triage_transcripts(ref.TEST_CASES)
    if not isinstance(res, list) or len(res) != len(ref.TEST_CASES):
        out["_note"] = f"triage_transcripts returned invalid length or type"
        return out

    match_count = 0
    for item, ref_case in zip(res, ref.TEST_CASES):
        if item.get("id") == ref_case["id"] and item.get("category") == ref_case["expected"]:
            match_count += 1

    out["triage_accuracy"] = float(match_count) / float(len(ref.TEST_CASES))
    if out["triage_accuracy"] < 1.0 and "_note" not in out:
        out["_note"] = f"Expected 100% triage accuracy, got {out['triage_accuracy']}"
    return out
