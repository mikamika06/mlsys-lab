import ref


def _rel_err(got, want):
    if want == 0.0:
        return 0.0 if got == 0.0 else float(abs(got))
    return float(abs(got - want) / abs(want))


def check(workdir):
    from memload.loader import compare_load_footprint

    out = {"footprint_rel_err": 0.0, "cases_passed": 0.0, "total_cases": float(len(ref.LOAD_CASES))}
    max_err = 0.0
    passed = 0

    for i, case in enumerate(ref.LOAD_CASES):
        want = ref.compare_load_footprint(case["tensors"], case["page_size"])
        try:
            got = compare_load_footprint(case["tensors"], case["page_size"])
        except Exception as e:
            out["_note"] = f"case {i} raised {type(e).__name__}: {e}"
            return out

        if not isinstance(got, dict):
            out["_note"] = f"case {i}: expected dict return, got {type(got)}"
            return out

        errs = []
        for key in ["heap_peak_bytes", "heap_resident_bytes", "mmap_virtual_bytes", "mmap_resident_bytes", "rss_savings_ratio"]:
            if key not in got:
                out["_note"] = f"case {i}: missing key '{key}'"
                return out
            e = _rel_err(float(got[key]), float(want[key]))
            errs.append(e)

        c_max_err = max(errs)
        max_err = max(max_err, c_max_err)
        if c_max_err <= 1e-3:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: max rel error {c_max_err:.6f}, got {got}, reference {want}"

    out["footprint_rel_err"] = float(max_err)
    out["cases_passed"] = float(passed)
    return out
