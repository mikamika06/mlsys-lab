import ref


def _rel_err(got, want):
    if want == 0.0:
        return 0.0 if got == 0.0 else float(abs(got))
    return float(abs(got - want) / abs(want))


def check(workdir):
    from memload.attribution import attribute_size_regression
    from memload.dedup import calculate_dedup_savings

    out = {
        "attribution_rel_err": 0.0,
        "dedup_rel_err": 0.0,
        "cases_passed": 0.0,
    }

    passed = 0
    max_attr_err = 0.0
    max_dedup_err = 0.0

    for i, case in enumerate(ref.ATTRIBUTION_CASES):
        want = ref.attribute_size_regression(case["base"], case["candidate"])
        try:
            got = attribute_size_regression(case["base"], case["candidate"])
        except Exception as e:
            out["_note"] = f"attribution case {i} raised {type(e).__name__}: {e}"
            return out

        if not isinstance(got, dict):
            out["_note"] = f"attribution case {i}: expected dict return"
            return out

        attr_errs = []
        for k in ["total_base_bytes", "total_candidate_bytes", "net_delta_bytes"]:
            if k not in got:
                out["_note"] = f"attribution case {i}: missing key '{k}'"
                return out
            attr_errs.append(_rel_err(float(got[k]), float(want[k])))

        if "category_deltas" not in got:
            out["_note"] = f"attribution case {i}: missing 'category_deltas'"
            return out
        for ck in ["added", "removed", "modified"]:
            if ck not in got["category_deltas"]:
                out["_note"] = f"attribution case {i}: missing category delta '{ck}'"
                return out
            attr_errs.append(_rel_err(float(got["category_deltas"][ck]), float(want["category_deltas"][ck])))

        if "by_layer" not in got or got["by_layer"] != want["by_layer"]:
            out["_note"] = f"attribution case {i}: by_layer mismatch"
            return out

        c_max_err = max(attr_errs)
        max_attr_err = max(max_attr_err, c_max_err)
        if c_max_err <= 1e-3:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"attribution case {i}: max error {c_max_err:.6f}"

    for i, case in enumerate(ref.DEDUP_CASES):
        want = ref.calculate_dedup_savings(case["tensors"], case["page_size"])
        try:
            got = calculate_dedup_savings(case["tensors"], case["page_size"])
        except Exception as e:
            out["_note"] = f"dedup case {i} raised {type(e).__name__}: {e}"
            return out

        if not isinstance(got, dict):
            out["_note"] = f"dedup case {i}: expected dict return"
            return out

        dedup_errs = []
        for k in ["raw_total_bytes", "unique_total_bytes", "disk_savings_bytes", "heap_savings_bytes", "mmap_savings_bytes", "heap_dedup_savings_ratio", "mmap_dedup_savings_ratio"]:
            if k not in got:
                out["_note"] = f"dedup case {i}: missing key '{k}'"
                return out
            dedup_errs.append(_rel_err(float(got[k]), float(want[k])))

        c_max_err = max(dedup_errs)
        max_dedup_err = max(max_dedup_err, c_max_err)
        if c_max_err <= 1e-3:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"dedup case {i}: max error {c_max_err:.6f}"

    out["attribution_rel_err"] = float(max_attr_err)
    out["dedup_rel_err"] = float(max_dedup_err)
    out["cases_passed"] = float(passed)
    return out
