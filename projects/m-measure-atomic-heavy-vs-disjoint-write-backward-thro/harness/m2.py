import ref


def check(workdir):
    out = {"throughput_ratio": 0.0}
    try:
        from triton_bw import measure
        ratio = measure.measure_throughput_ratio()
        out["throughput_ratio"] = float(ratio)
        if ratio < 0.8:
            out["_note"] = f"throughput ratio {ratio} is below threshold 0.8"
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
