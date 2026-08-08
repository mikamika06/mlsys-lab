import ref


def check(workdir):
    from compilebench.bench import compare_latency

    out = {
        "configs_matched": 0.0,
        "configs": float(len(ref.CONFIGS)),
        "latency_ratio_correct": 0.0,
    }
    matched = 0
    ratio_ok = True

    for i, cfg in enumerate(ref.CONFIGS):
        seq = ref.REQUEST_SEQUENCES[i]
        want = ref.compare_latency(seq, cfg)
        got = compare_latency(seq, cfg)

        keys = ["compile_latencies", "export_latencies", "recompile_count", "max_spike_ratio", "total_latency_ratio"]
        if all(k in got for k in keys):
            if (
                got["recompile_count"] == want["recompile_count"]
                and abs(got["max_spike_ratio"] - want["max_spike_ratio"]) < 1e-4
                and abs(got["total_latency_ratio"] - want["total_latency_ratio"]) < 1e-4
            ):
                matched += 1
            else:
                ratio_ok = False
                if "_note" not in out:
                    out["_note"] = f"config {i}: got spike ratio {got.get('max_spike_ratio')}, want {want['max_spike_ratio']}"
        else:
            ratio_ok = False
            if "_note" not in out:
                out["_note"] = f"config {i}: missing response keys"

    out["configs_matched"] = float(matched)
    out["latency_ratio_correct"] = 1.0 if (matched == len(ref.CONFIGS) and ratio_ok) else 0.0
    return out
