import ref


def check(workdir):
    from ortperf.memcpy import locate_boundary_memcpys
    from ortperf.overhead import compute_overhead

    out = {"memcpy_matched": 0.0, "overhead_matched": 0.0}

    m_ok = 0
    for i, p in enumerate(ref.PROFILES):
        want_m = ref.locate_boundary_memcpys(p)
        got_m = locate_boundary_memcpys(p)
        if got_m == want_m:
            m_ok += 1
        elif "_note" not in out:
            out["_note"] = f"memcpy profile {i}: got {got_m}, want {want_m}"
    if m_ok == len(ref.PROFILES):
        out["memcpy_matched"] = 1.0

    o_ok = 0
    for i, p in enumerate(ref.PROFILES):
        want_o = ref.compute_overhead(p)
        got_o = compute_overhead(p)
        if abs(got_o - want_o) < 1e-5:
            o_ok += 1
        elif "_note" not in out:
            out["_note"] = f"overhead profile {i}: got {got_o}, want {want_o}"
    if o_ok == len(ref.PROFILES):
        out["overhead_matched"] = 1.0

    return out
