import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    try:
        from bench_serve.bundle import create_result_bundle
        from bench_serve.runner import run_multi_rate_bench
    except ImportError as e:
        return {
            "bundle_valid": 0.0,
            "summary_matched": 0.0,
            "_note": f"Import error: {e}",
        }

    oracle_raw = ref.oracle_run_multi_rate_bench(ref.REQUEST_DATA, ref.RATES)
    oracle_bundle = ref.oracle_create_result_bundle("test-model", oracle_raw)

    try:
        got_raw = run_multi_rate_bench(ref.REQUEST_DATA, ref.RATES)
        got_bundle = create_result_bundle("test-model", got_raw)
    except Exception as e:
        return {
            "bundle_valid": 0.0,
            "summary_matched": 0.0,
            "_note": f"Execution error: {e}",
        }

    bundle_valid = 0.0
    summary_matched = 0.0

    if (
        isinstance(got_bundle, dict)
        and "rates" in got_bundle
        and "summary" in got_bundle
    ):
        if len(got_bundle["rates"]) == len(ref.RATES):
            bundle_valid = 1.0

    if bundle_valid == 1.0:
        got_max_tp = got_bundle["summary"].get("max_throughput", -1.0)
        want_max_tp = oracle_bundle["summary"]["max_throughput"]
        if abs(got_max_tp - want_max_tp) < 1e-3:
            summary_matched = 1.0
        else:
            note = f"max_throughput mismatch: got {got_max_tp}, want {want_max_tp}"
            return {
                "bundle_valid": bundle_valid,
                "summary_matched": 0.0,
                "_note": note,
            }

    return {"bundle_valid": bundle_valid, "summary_matched": summary_matched}
