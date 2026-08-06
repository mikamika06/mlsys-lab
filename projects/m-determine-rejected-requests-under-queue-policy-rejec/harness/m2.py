import ref

def check(workdir):
    from batcher.diagnose import attribute_latency
    out = {"attribution_matched": 0.0}
    ok = 0
    for sc in ref.ATTRIBUTIONS:
        got = attribute_latency(
            sc["b_wait"], sc["b_exec"], sc["c_wait"], sc["c_exec"]
        )
        if got == sc["want"]:
            ok += 1
    out["attribution_matched"] = float(ok)
    return out
