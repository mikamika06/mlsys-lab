import ref


def check(workdir):
    from autotune.overhead import compute_overhead_reduction

    out = {"overhead_reduced_matched": 0.0}
    ok = 0
    for i, (b, a) in enumerate(ref.TRACES):
        b_tot = sum(item.get("launch_delay", 0) + item.get("driver_wait", 0) for item in b)
        a_tot = sum(item.get("launch_delay", 0) + item.get("driver_wait", 0) for item in a)
        want_val = float(b_tot - a_tot) / float(b_tot) if b_tot > 0 else 1.0

        try:
            got = compute_overhead_reduction(b, a)
            if abs(float(got) - float(want_val)) < 1e-5:
                ok += 1
        except Exception:
            pass
    out["overhead_reduced_matched"] = float(ok)
    return out
