import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from cascade.benchmark import compute_latency_ratio, run_cascade_benchmark

    out = {"ratios_matched": 0.0, "latency_ratio": 1.0}

    cfg = ref.CONFIGS[2]
    ref_res = ref.build_reference_results()[2]

    try:
        got_res = run_cascade_benchmark(cfg)
        calc_ratio = compute_latency_ratio(got_res["cascade_latency"], got_res["single_stage_latency"])

        if abs(calc_ratio - ref_res["latency_ratio"]) < 1e-5:
            out["ratios_matched"] = 1.0

        out["latency_ratio"] = float(got_res["latency_ratio"])
    except Exception as e:
        out["_note"] = f"m2 check failed with exception: {type(e).__name__}: {str(e)}"

    return out
