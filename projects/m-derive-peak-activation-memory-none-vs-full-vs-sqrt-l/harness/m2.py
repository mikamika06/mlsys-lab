import ref


def check(workdir):
    from ckpt import analysis

    out = {"overhead_cases_matched": 0.0}
    ok = 0
    for layers, segment_size in ref.TEST_CASES_OVERHEAD:
        want = ref.recompute_flops_overhead(layers, segment_size)
        try:
            got = analysis.recompute_flops_overhead(layers, segment_size)
        except Exception as e:
            out["_note"] = f"raised {type(e).__name__} for layers={layers}, seg={segment_size}"
            return out
        if got is not None and abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch for layers={layers}, seg={segment_size}: got {got}, want {want}"

    out["overhead_cases_matched"] = float(ok)
    return out
