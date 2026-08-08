import ref


def check(workdir):
    from triton_bench.metrics import evaluate_throughput

    out = {"throughput_ratio": 0.0}
    try:
        ratio = evaluate_throughput(ref.FUSED_MS, ref.UNFUSED_MS)
        out["throughput_ratio"] = float(ratio)
    except Exception as e:
        out["_note"] = f"error: {str(e)[:100]}"
    return out
