import ref


def check(workdir):
    from decoder.pipeline import measure_latencies
    from decoder.metrics import analyze_latency_ratio

    model, tokenizer, prompt = ref.get_mock_setup()
    cold, reused = measure_latencies(model, tokenizer, prompt)
    ratio = analyze_latency_ratio(cold, reused)

    out = {"speedup_ratio": float(ratio)}
    if ratio < 1.2:
        out["_note"] = f"Expected speedup ratio >= 1.2, got {ratio}"
    return out
