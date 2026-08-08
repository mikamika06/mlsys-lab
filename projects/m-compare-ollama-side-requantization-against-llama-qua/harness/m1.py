import ref


def check(workdir):
    from ollamabridge.quant import compare_requantization
    out = {"quant_matched": 0.0}
    try:
        res = compare_requantization({}, "Q4_K_M")
        if isinstance(res, dict) and "ollama_error" in res and "llama_error" in res:
            out["quant_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:100]}"
    return out
