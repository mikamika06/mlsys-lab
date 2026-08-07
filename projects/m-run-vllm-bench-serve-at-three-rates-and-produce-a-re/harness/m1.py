import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    try:
        from bench_serve.runner import run_multi_rate_bench
    except ImportError as e:
        return {
            "throughput_ratio": 0.0,
            "rates_completed": 0,
            "_note": f"Import error: {e}",
        }

    oracle_data = ref.oracle_run_multi_rate_bench(ref.REQUEST_DATA, ref.RATES)
    try:
        got_data = run_multi_rate_bench(ref.REQUEST_DATA, ref.RATES)
    except Exception as e:
        return {
            "throughput_ratio": 0.0,
            "rates_completed": 0,
            "_note": f"Execution error: {e}",
        }

    rates_completed = 0
    tps_got = []
    tps_want = []

    for rate in ref.RATES:
        if rate in got_data:
            rates_completed += 1
            ref_res = ref.oracle_calculate_metrics(
                oracle_data[rate]["results"], oracle_data[rate]["duration"]
            )
            got_res = ref.oracle_calculate_metrics(
                got_data[rate]["results"], got_data[rate]["duration"]
            )
            tps_want.append(ref_res["total_throughput_tok_s"])
            tps_got.append(got_res["total_throughput_tok_s"])

    if not tps_want or sum(tps_want) == 0:
        ratio = 0.0
    else:
        ratio = min(sum(tps_got) / sum(tps_want), 1.0)

    return {
        "throughput_ratio": float(ratio),
        "rates_completed": rates_completed,
    }
