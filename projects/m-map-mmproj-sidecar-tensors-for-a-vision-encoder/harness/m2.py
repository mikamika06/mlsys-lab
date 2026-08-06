import ref

def check(workdir):
    from mmproj.mapping import map_tensors

    raw = ref.get_raw_names(layers=2)
    want = ref.map_tensors(raw)
    try:
        got = map_tensors(raw)
    except Exception as e:
        return {"exact_match": 0.0, "_note": f"crashed: {e}"}

    if got == want:
        return {"exact_match": 1.0}

    missing = set(want.keys()) - set((got or {}).keys())
    bad = {k: got[k] for k in want if k in got and got[k] != want[k]}
    note = ""
    if missing:
        note += f"missing keys like {list(missing)[0]}. "
    if bad:
        k = list(bad.keys())[0]
        note += f"mismatch on {k}: got {bad[k]}, want {want[k]}."

    return {"exact_match": 0.0, "_note": note}
