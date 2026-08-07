import ref

def check(workdir):
    from qfusion.placement import insert_per_channel_qdq
    matched = 0
    total = len(ref.CASES_M2)
    for case in ref.CASES_M2:
        want = ref.insert_qdq(case)
        got = insert_per_channel_qdq(case)
        if got == want:
            matched += 1
    score = float(matched == total)
    out = {"placement_matched": score, "total": float(total), "matched": float(matched)}
    if score < 1.0:
        out["_note"] = f"Placed {matched}/{total} per-channel Q/DQ correctly."
    return out
