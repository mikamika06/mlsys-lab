import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from cascade.benchmark import run_cascade_benchmark

    out = {"benchmarks_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    ref_results = ref.build_reference_results()

    for i, cfg in enumerate(ref.CONFIGS):
        ref_res = ref_results[i]
        try:
            got_res = run_cascade_benchmark(cfg)
            if (got_res["accepted_count"] == ref_res["accepted_count"] and
                    got_res["draft_count"] == ref_res["draft_count"] and
                    abs(got_res["cascade_latency"] - ref_res["cascade_latency"]) < 1e-5):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got_res}, reference {ref_res}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised exception: {type(e).__name__}: {str(e)}"

    out["benchmarks_matched"] = float(ok)
    return out
