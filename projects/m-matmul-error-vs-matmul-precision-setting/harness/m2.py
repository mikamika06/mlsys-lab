import ref

def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"unsafe_layers_match": 0.0}
    try:
        import numerics
        want = ref.find_unsafe_layers(ref.LAYERS, ref.LAYER_INPUT, 0.05)
        got = numerics.find_unsafe_layers(ref.LAYERS, ref.LAYER_INPUT, 0.05)
        if got == want:
            out["unsafe_layers_match"] = 1.0
        else:
            out["_note"] = f"Expected unsafe indices {want}, got {got}"
    except Exception as e:
        out["_note"] = str(e)
    return out
