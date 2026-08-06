import ref

def check(workdir):
    from flashbuild.cost import latency_ratio
    out = {"latency_ratio": 0.0}
    c1 = ref.CONFIGS[1]
    c2 = ref.CONFIGS[0]
    want = ref.compute_latency_ratio(c1, c2)
    try:
        got = latency_ratio(c1, c2)
        if isinstance(got, (int, float)) and got > 0:
            out["latency_ratio"] = float(got)
    except Exception:
        pass
    return out
