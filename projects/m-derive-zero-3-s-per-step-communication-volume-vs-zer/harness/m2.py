import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from zerocheck.parser import parse_memory_reduction

    tc = ref.get_test_cases()
    got = parse_memory_reduction(tc["log_lines"])
    want = {
        0: {"zero1": 850000000, "zero3": 320000000, "reduction_pct": (850000000 - 320000000) / 850000000 * 100.0},
        1: {"zero1": 850000000, "zero3": 320000000, "reduction_pct": (850000000 - 320000000) / 850000000 * 100.0}
    }

    out = {"reduction_matched": 0.0}
    if got == want:
        out["reduction_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
