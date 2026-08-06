import ref

def check(workdir):
    from perfanalysis.latency import attribute_delta
    out = {"latency_matched": 0.0}
    runs = [
        ({"queue_ms": 1.0, "compute_ms": 10.0, "total_ms": 11.0}, {"queue_ms": 5.0, "compute_ms": 12.0, "total_ms": 17.0}),
        ({"queue_ms": 2.0, "compute_ms": 20.0, "total_ms": 22.0}, {"queue_ms": 10.0, "compute_ms": 15.0, "total_ms": 25.0}),
        ({"queue_ms": 0.5, "compute_ms": 5.0, "total_ms": 5.5}, {"queue_ms": 2.0, "compute_ms": 6.0, "total_ms": 8.0})
    ]
    matched = 0
    for off, on in runs:
        want = ref.attribute_delta(off, on)
        got = attribute_delta(off, on)
        if all(abs(got[k] - want[k]) < 1e-5 for k in want):
            matched += 1
    out["latency_matched"] = float(matched)
    return out
