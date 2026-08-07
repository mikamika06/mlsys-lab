import ref

def check(workdir):
    from repack.predict import predict_variant
    out = {"variants_matched": 0.0, "total": float(len(ref.TEST_CASES))}
    ok = 0
    for i, (features, expected) in enumerate(ref.TEST_CASES):
        got = predict_variant(features)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, expected {expected}"
    out["variants_matched"] = float(ok)
    return out
