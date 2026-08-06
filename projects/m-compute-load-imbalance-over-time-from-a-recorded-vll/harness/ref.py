import numpy as np


def generate_benchmark_data():
    np.random.seed(42)
    num_timestamps = 20
    log_entries = []

    for t in range(num_timestamps):
        raw = np.random.randint(50, 500, size=8).tolist()
        eplb = (np.array(raw) * 0.8 + 50).tolist()
        log_entries.append({
            "timestamp": float(1000 + t * 5),
            "raw_expert_tokens": raw,
            "eplb_expert_tokens": eplb,
        })

    benchmark_rows = [
        {"ep_degree": 1, "tokens_per_sec": 1200.0},
        {"ep_degree": 2, "tokens_per_sec": 2100.0},
        {"ep_degree": 4, "tokens_per_sec": 3800.0},
        {"ep_degree": 8, "tokens_per_sec": 6900.0},
    ]
    target_ep_degrees = [16, 32, 64]

    vllm_trace = []
    sglang_trace = []
    for _ in range(30):
        vllm_trace.append({
            "dispatch_latency_ms": float(np.random.uniform(1.2, 2.5)),
            "exec_latency_ms": float(np.random.uniform(15.0, 25.0)),
        })
        sglang_trace.append({
            "dispatch_latency_ms": float(np.random.uniform(0.8, 1.8)),
            "exec_latency_ms": float(np.random.uniform(14.0, 23.0)),
        })

    return {
        "log_entries": log_entries,
        "benchmark_rows": benchmark_rows,
        "target_ep_degrees": target_ep_degrees,
        "vllm_trace": vllm_trace,
        "sglang_trace": sglang_trace,
    }


def compute_imbalance_over_time(log_entries):
    timestamps = []
    imbalance_ratios = []
    eplb_effective_ratios = []

    for entry in log_entries:
        ts = float(entry["timestamp"])
        raw_tokens = np.array(entry["raw_expert_tokens"], dtype=np.float64)
        eplb_tokens = np.array(entry["eplb_expert_tokens"], dtype=np.float64)

        raw_max = np.max(raw_tokens)
        raw_mean = np.mean(raw_tokens)
        raw_ratio = float(raw_max / raw_mean) if raw_mean > 0 else 1.0

        eplb_max = np.max(eplb_tokens)
        eplb_mean = np.mean(eplb_tokens)
        eplb_ratio = float(eplb_max / eplb_mean) if eplb_mean > 0 else 1.0

        timestamps.append(ts)
        imbalance_ratios.append(raw_ratio)
        eplb_effective_ratios.append(eplb_ratio)

    return {
        "timestamps": timestamps,
        "imbalance_ratios": imbalance_ratios,
        "eplb_effective_ratios": eplb_effective_ratios,
        "mean_imbalance": float(np.mean(imbalance_ratios)) if imbalance_ratios else 0.0,
    }


def fit_and_extrapolate_throughput(benchmark_rows, target_ep_degrees):
    ep_degrees = np.array([r["ep_degree"] for r in benchmark_rows], dtype=np.float64)
    throughputs = np.array([r["tokens_per_sec"] for r in benchmark_rows], dtype=np.float64)

    log_ep = np.log(ep_degrees)
    log_tp = np.log(throughputs)

    poly = np.polyfit(log_ep, log_tp, 1)
    scaling_exponent = float(poly[0])
    intercept = float(poly[1])

    extrapolated = {}
    for ep in target_ep_degrees:
        pred_log_tp = scaling_exponent * np.log(float(ep)) + intercept
        extrapolated[int(ep)] = float(np.exp(pred_log_tp))

    return {
        "scaling_exponent": scaling_exponent,
        "intercept": intercept,
        "extrapolated_throughput": extrapolated,
    }


def compare_serving_traces(vllm_trace, sglang_trace):
    v_dispatch = np.array([r["dispatch_latency_ms"] for r in vllm_trace], dtype=np.float64)
    v_exec = np.array([r["exec_latency_ms"] for r in vllm_trace], dtype=np.float64)

    s_dispatch = np.array([r["dispatch_latency_ms"] for r in sglang_trace], dtype=np.float64)
    s_exec = np.array([r["exec_latency_ms"] for r in sglang_trace], dtype=np.float64)

    v_total = v_dispatch + v_exec
    s_total = s_dispatch + s_exec

    v_avg_total = float(np.mean(v_total))
    s_avg_total = float(np.mean(s_total))

    latency_ratio = v_avg_total / s_avg_total if s_avg_total > 0 else 1.0
    dispatch_diff_ms = float(np.mean(v_dispatch) - np.mean(s_dispatch))

    return {
        "vllm_avg_latency_ms": v_avg_total,
        "sglang_avg_latency_ms": s_avg_total,
        "latency_ratio_vllm_vs_sglang": float(latency_ratio),
        "dispatch_difference_ms": dispatch_diff_ms,
    }
