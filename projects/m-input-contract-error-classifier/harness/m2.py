import ref

def check(workdir):
    try:
        from flash_contract.classifier import classify_inputs
    except ImportError:
        return {"matches": 0.0, "_note": "failed to import classify_inputs"}

    ok = 0
    total = len(ref.CASES)
    out = {}

    for i, (q, k, v) in enumerate(ref.CASES):
        try:
            got = classify_inputs(q, k, v)
            want = ref.classify_inputs(q, k, v)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} crashed: {e}"

    out["matches"] = 1.0 if ok == total else ok / float(total)
    return out
