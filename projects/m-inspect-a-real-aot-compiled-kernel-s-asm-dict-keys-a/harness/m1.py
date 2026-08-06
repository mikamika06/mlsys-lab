import ref

def check(workdir):
    try:
        from inspector.asm import analyze_asm_dict
    except ImportError:
        return {"matches_ref": 0.0, "_note": "Could not import analyze_asm_dict"}

    out = {"matches_ref": 0.0}
    ok = 0
    for i, asm in enumerate(ref.ASM_DICTS):
        want = ref.analyze_asm_dict(asm)
        try:
            got = analyze_asm_dict(asm)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"case {i}: expected {want}, got {got}"
        except Exception as e:
            out["_note"] = f"error on case {i}: {e}"

    out["matches_ref"] = float(ok == len(ref.ASM_DICTS))
    return out
