import ref

def check(workdir):
    from fsdpfit.sizing import calculate_min_world_size
    out = {"sizing_matched": 0.0}
    ok = 0
    for p, o, g, a, b in ref.CONFIGS:
        want = ref.calculate_min_world_size(p, o, g, a, b)
        got = calculate_min_world_size(p, o, g, a, b)
        if got == want:
            ok += 1
    out["sizing_matched"] = 1.0 if ok == len(ref.CONFIGS) else 0.0
    return out
