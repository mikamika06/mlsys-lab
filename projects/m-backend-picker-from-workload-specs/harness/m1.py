import ref


def check(workdir):
    from picker.selector import select_backend

    out = {"workloads_matched": 0.0}
    ok = 0
    total = len(ref.WORKLOADS)
    for i, spec in enumerate(ref.WORKLOADS):
        want = ref.select_backend(spec)
        try:
            got = select_backend(spec)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"workload {i}: expected {want}, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"workload {i} raised {type(e).__name__}: {e}"
    if ok == total:
        out["workloads_matched"] = 1.0
    return out
