import ref


def check(workdir):
    from efficiency.bench import measure_step_latencies

    out = {"latency_ratio_valid": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        got = measure_step_latencies(cfg)
        expected_ft = sum(cfg["steps_ft"]) / len(cfg["steps_ft"])
        expected_lora = sum(cfg["steps_lora"]) / len(cfg["steps_lora"])
        expected_ratio = expected_ft / expected_lora

        if "latency_ratio" in got and abs(got["latency_ratio"] - expected_ratio) < 1e-5:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["latency_ratio_valid"] = 1.0
    else:
        out["_note"] = f"Expected latency ratio computation to match reference for all configs, got {ok}/{len(ref.CONFIGS)}"
    return out
