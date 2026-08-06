import ref

def check(workdir):
    from solparser.diff import diff_basic_full
    basic_csv, full_csv = ref.generate_basic_and_full_csv()

    try:
        got = diff_basic_full(basic_csv, full_csv)
    except Exception as e:
        return {"sections_matched": 0.0, "_note": f"Exception raised: {e}"}

    expected = ["LaunchStats", "MemoryWorkloadAnalysis"]

    matched = 1.0 if sorted(got) == sorted(expected) else 0.0
    out = {"sections_matched": matched}
    if matched == 0.0:
        out["_note"] = f"Expected {expected}, got {got}"
    return out
