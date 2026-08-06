import ref

def check(workdir):
    try:
        from inspector.stages import compare_num_stages
        from inspector.portability import classify_snippet
    except ImportError:
        return {"stages_matches": 0.0, "classify_matches": 0.0, "_note": "Could not import functions"}

    out = {"stages_matches": 0.0, "classify_matches": 0.0}

    want_stages = ref.compare_num_stages(ref.ASM_DICT_2, ref.ASM_DICT_4)
    try:
        got_stages = compare_num_stages(ref.ASM_DICT_2, ref.ASM_DICT_4)
        if got_stages == want_stages:
            out["stages_matches"] = 1.0
        else:
            out["_note"] = f"stages: expected {want_stages}, got {got_stages}"
    except Exception as e:
        out["_note"] = f"stages error: {e}"

    ok = 0
    for s in ref.SNIPPETS:
        try:
            if classify_snippet(s) == ref.classify_snippet(s):
                ok += 1
        except Exception:
            pass

    if ok == len(ref.SNIPPETS):
        out["classify_matches"] = 1.0

    return out
