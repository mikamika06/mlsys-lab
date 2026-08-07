import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from zero3.formula import zero3_memory_math
    except ImportError:
        return {"_note": "could not import zero3_memory_math"}

    out = {"math_matched": 0.0, "math_total": float(len(ref.CONFIGS))}
    ok = 0
    for layers, gpus in ref.CONFIGS:
        want = ref.zero3_memory_math(layers, gpus)
        try:
            got = zero3_memory_math(layers, gpus)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"mismatch: got {got}, want {want}"
        except Exception as e:
            out["_note"] = f"exception: {e}"
            break
    out["math_matched"] = float(ok)
    sys.path.pop(0)
    return out
