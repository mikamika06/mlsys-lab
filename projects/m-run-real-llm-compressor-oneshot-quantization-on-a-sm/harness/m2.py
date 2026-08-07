import ref

def check(workdir):
    from compressor_workflow.metrics import measure_metrics
    out = {"ratio_matched": 0.0, "perplexity_matched": 0.0}

    orig = 1048576
    quant = 262144
    stub = "model-stub"
    data = [1, 2, 3]

    try:
        metrics = measure_metrics(orig, quant, stub, data)
    except Exception as e:
        out["_note"] = f"measure_metrics failed: {e}"
        return out

    want_ratio = ref.compute_compression_ratio(orig, quant)
    want_ppl = ref.compute_perplexity(stub, data)

    if metrics and abs(metrics.get("compression_ratio", 0) - want_ratio) < 1e-5:
        out["ratio_matched"] = 1.0
    else:
        out["_note"] = f"compression_ratio mismatch: got {metrics.get('compression_ratio')}, want {want_ratio}"

    if metrics and abs(metrics.get("perplexity", 0) - want_ppl) < 1e-5:
        out["perplexity_matched"] = 1.0
    else:
        out["_note"] = f"perplexity mismatch: got {metrics.get('perplexity')}, want {want_ppl}"

    return out
