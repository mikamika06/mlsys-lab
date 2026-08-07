import ref

def check(workdir):
    m = {"identical": 0.0}
    try:
        tokens = [101, 2056, 2003, 1037]
        out1 = ref.run_cached_vs_uncached(tokens)
        out2 = ref.run_cached_vs_uncached(tokens)
        if len(out1) == len(out2) and all(a == b for a, b in zip(out1, out2)):
            m["identical"] = 1.0
    except Exception:
        pass
    return m
