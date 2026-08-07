import ref


def check(workdir):
    from fp8quant.pipeline import run_compression
    out = {"size_ratio": 1.0}
    try:
        ratio = run_compression()
        out["size_ratio"] = float(ratio)
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
