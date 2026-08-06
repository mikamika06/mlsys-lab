import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from moeload.benchmark import run_benchmark_session
    from moeload.metrics import calculate_summary, compute_latency_degradation_ratio
    from moeload.server import MoEServer

    out = {"metrics_matched": 0.0, "latency_ratio": 0.0}

    server = MoEServer()
    ref_server = ref.RefMoEServer()

    workload = ref.WORKLOADS[2]

    low_traces = run_benchmark_session(server, workload, concurrency=1)
    high_traces = run_benchmark_session(server, workload, concurrency=8)

    ref_low_traces = ref.ref_run_benchmark_session(ref_server, workload, concurrency=1)
    ref_high_traces = ref.ref_run_benchmark_session(ref_server, workload, concurrency=8)

    got_low = calculate_summary(low_traces)
    got_high = calculate_summary(high_traces)

    want_low = ref.ref_calculate_summary(ref_low_traces)
    want_high = ref.ref_calculate_summary(ref_high_traces)

    keys = ["throughput_tok_per_sec", "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_ttft_ms"]
    match_low = all(abs(got_low.get(k, 0.0) - want_low.get(k, 0.0)) < 1e-3 for k in keys)
    match_high = all(abs(got_high.get(k, 0.0) - want_high.get(k, 0.0)) < 1e-3 for k in keys)

    if match_low and match_high:
        out["metrics_matched"] = 1.0
    else:
        out["_note"] = f"Metrics mismatch: got_high={got_high}, want_high={want_high}"

    ratio = compute_latency_degradation_ratio(got_low, got_high)
    out["latency_ratio"] = float(ratio)

    return out
