import ref


def check(workdir):
    from finetune.fragmentation import analyze_memory_summary

    out = {"ratio_match": 0.0, "severity_match": 0.0}
    sample_summary = "Allocated memory: 1000000 bytes\nReserved memory: 1350000 bytes"
    want = ref.analyze_memory_summary(sample_summary)
    try:
        got = analyze_memory_summary(sample_summary)
    except Exception as e:
        out["_note"] = f"analyze_memory_summary raised {type(e).__name__}: {str(e)[:120]}"
        return out

    if not isinstance(got, dict):
        out["_note"] = f"expected dict, got {type(got)}"
        return out

    r_want = want.get("fragmentation_ratio")
    r_got = got.get("fragmentation_ratio")
    if r_want is not None and r_got is not None and abs(r_want - r_got) < 1e-3:
        out["ratio_match"] = 1.0
    else:
        out["_note"] = f"ratio mismatch: got {r_got}, reference {r_want}"

    s_want = want.get("severity")
    s_got = got.get("severity")
    if s_want == s_got:
        out["severity_match"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"severity mismatch: got {s_got}, reference {s_want}"

    return out
