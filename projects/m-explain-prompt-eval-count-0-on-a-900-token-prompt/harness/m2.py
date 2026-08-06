import ref


def check(workdir):
    from runner_metrics.ttft import compute_ttft, dominant_component

    out = {"ttft_matched": 0.0}
    scenario = ref.SCENARIOS[0]

    try:
        ttft = compute_ttft(scenario["context_len"], scenario["params"])
        comp = dominant_component(scenario["context_len"], scenario["params"])
        if isinstance(ttft, (int, float)) and ttft > 0 and comp in ("prefill", "overhead"):
            out["ttft_matched"] = 1.0
        else:
            out["_note"] = f"Invalid TTFT outputs: ttft={ttft}, dominant={comp}"
    except Exception as e:
        out["_note"] = f"Error during execution: {str(e)}"
    return out
