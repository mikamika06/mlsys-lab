def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.analyzer import compute_bpw_and_size
    import ref

    m = {"bpw_calculated": 0.0, "file_size_correct": 0.0}
    try:
        p, s = compute_bpw_and_size(7000000000, 4.0)
        op, osz = ref.oracle_compute_bpw_and_size(7000000000, 4.0)
        if abs(p - op) < 1e-5:
            m["bpw_calculated"] = 1.0
        if s == osz:
            m["file_size_correct"] = 1.0
    except Exception:
        pass
    return m
