import ref

def check(workdir):
    from packutil.metrics import compute_utilization
    lengths = [128, 256, 64, 512, 100]
    max_len = 512
    want = ref.compute_utilization(lengths, max_len)
    got = compute_utilization(lengths, max_len)
    match = 1.0 if abs(want["packing_utilization"] - got["packing_utilization"]) < 1e-5 and abs(want["padding_utilization"] - got["padding_utilization"]) < 1e-5 else 0.0
    out = {"utilization_matched": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
