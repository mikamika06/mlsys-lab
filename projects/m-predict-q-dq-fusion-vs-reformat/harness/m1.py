import ref

def check(workdir):
    from qfusion.fusion import predict_fusion
    matched = 0
    total = len(ref.CASES_M1)
    for case in ref.CASES_M1:
        got = predict_fusion(case)
        if got == case["expected"]:
            matched += 1
    score = float(matched == total)
    out = {"predictions_matched": score, "total": float(total), "matched": float(matched)}
    if score < 1.0:
        out["_note"] = f"Matched {matched}/{total} fusion predictions correctly."
    return out
