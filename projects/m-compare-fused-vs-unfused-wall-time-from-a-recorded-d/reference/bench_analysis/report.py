from bench_analysis.metrics import compute_latency_stats, compute_throughput_gbs
from bench_analysis.parser import extract_tensor_bytes, parse_do_bench_trace


def generate_benchmark_summary(record):
    unfused_stats = parse_do_bench_trace(record["unfused_trace"])
    fused_stats = parse_do_bench_trace(record["fused_trace"])

    elem_bytes = extract_tensor_bytes(record["shape"], record["dtype"])
    num_inputs = record.get("num_inputs", 2)
    num_outputs = record.get("num_outputs", 1)
    num_ops = record.get("num_unfused_ops", 2)

    unfused_io_bytes = elem_bytes * (num_inputs + num_outputs + (num_ops - 1))
    fused_io_bytes = elem_bytes * (num_inputs + num_outputs)

    latency = compute_latency_stats(
        unfused_stats["mean_ms"], fused_stats["mean_ms"]
    )

    unfused_bw = compute_throughput_gbs(
        unfused_stats["mean_ms"], unfused_io_bytes
    )
    fused_bw = compute_throughput_gbs(fused_stats["mean_ms"], fused_io_bytes)

    return {
        "unfused_mean_ms": unfused_stats["mean_ms"],
        "fused_mean_ms": fused_stats["mean_ms"],
        "speedup": latency["speedup"],
        "time_saved_ms": latency["time_saved_ms"],
        "unfused_gbps": unfused_bw,
        "fused_gbps": fused_bw,
        "throughput_ratio": (
            fused_bw / unfused_bw if unfused_bw > 0 else 0.0
        ),
    }
