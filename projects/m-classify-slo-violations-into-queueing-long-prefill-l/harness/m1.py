import ref

def check(workdir):
    from sloclassify.parser import parse_request
    out = {"timings_matched": 0.0}
    ok = 0
    for r in ref.REQUESTS:
        want = ref.parse_request(r)
        got = parse_request(r)
        if got and all(abs(got.get(k, 0) - want[k]) < 1e-5 for k in ["queue_time", "prefill_time", "output_time", "total_latency"]):
            ok += 1
    out["timings_matched"] = float(ok)
    return out
