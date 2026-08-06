import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from effbpw.compute import compute_effective_bpw
    except ImportError:
        return {"rel_err": 1.0, "_note": "failed to import compute_effective_bpw"}

    out = {"rel_err": 0.0}
    max_err = 0.0

    for t_shapes, quants, _, _, _, _, _ in ref.FIXTURES:
        for q_name, base_bpw in quants.items():
            want = ref.compute_effective_bpw(t_shapes, base_bpw)
            try:
                got = compute_effective_bpw(t_shapes, base_bpw)
            except NotImplementedError:
                return {"rel_err": 1.0, "_note": "NotImplementedError raised"}

            err = abs(want - got) / max(1e-9, want)
            max_err = max(max_err, err)

    out["rel_err"] = float(max_err)
    return out
