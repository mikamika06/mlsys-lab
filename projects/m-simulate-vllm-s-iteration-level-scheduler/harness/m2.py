import ref

def check(workdir):
    from vllmsched.metrics import measure_throughput
    levels = [1, 2, 4, 8, 16, 32, 64]
    want = ref.measure_throughput(levels)
    try:
        got = measure_throughput(levels)
        match = 1.0 if len(got) == len(want) and all(abs(a - b) < 1e-5 for a, b in zip(got, want)) else 0.0
    except Exception:
        match = 0.0
    out = {"throughput_matched": match}
    if match == 0.0:
        out["_note"] = "throughput curve values do not match expected reference model"
    return out
