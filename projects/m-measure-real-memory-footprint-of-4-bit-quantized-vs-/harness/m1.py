import ref

def check(workdir):
    from qlora_mem.measure import measure_footprint
    t = ref.generate_tensor()
    want = ref.measure_footprint(t)
    got = measure_footprint(t)
    out = {"footprint_matched": 0.0}
    if got == want:
        out["footprint_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"
    return out
