import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from zerocheck.analysis import compute_communication_volumes

    tc = ref.get_test_cases()
    got = compute_communication_volumes(tc["numel"], tc["bytes_per_elem"], tc["world_size"])

    total_bytes = tc["numel"] * tc["bytes_per_elem"]
    ws = tc["world_size"]
    want_z1 = 2.0 * total_bytes * (ws - 1) / ws
    want_z3 = 2.0 * total_bytes * (ws - 1) / ws + total_bytes

    out = {"volumes_matched": 0.0}
    if abs(got.get("zero1", 0) - want_z1) < 1e-5 and abs(got.get("zero3", 0) - want_z3) < 1e-5:
        out["volumes_matched"] = 1.0
    else:
        out["_note"] = f"expected zero1={want_z1}, zero3={want_z3}, got {got}"
    return out
