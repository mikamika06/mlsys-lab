import ref

def check(workdir):
    from prof.analysis import compare_execution
    traces, _ = ref.generate_fixtures()
    ok = 0
    out = {"comparison_matched": 0.0, "total": float(len(traces))}
    for i, (e, c) in enumerate(traces):
        got = compare_execution(e, c)
        e_ops = e.get("ops", 0)
        c_ops = c.get("ops", 0)
        e_size = e.get("size", 0)
        c_size = c.get("size", 0)
        want_size_ratio = c_size / (e_size if e_size > 0 else 1.0)
        want_op_ratio = c_ops / (e_ops if e_ops > 0 else 1.0)
        if abs(got.get("size_ratio", 0) - want_size_ratio) < 1e-5 and abs(got.get("op_ratio", 0) - want_op_ratio) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch at index {i}"
    out["comparison_matched"] = 1.0 if ok == len(traces) else 0.0
    return out
