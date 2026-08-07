import ref


def check(workdir):
    from latency.detector import classify_run

    out = {"fallback_flagged": 0.0, "reassociation_handled": 0.0}

    fallback_cfg = ref.CONFIGS[2]
    got_fallback = classify_run(
        fallback_cfg["samples"],
        fallback_cfg["baseline_med"],
        fallback_cfg["baseline_mad"],
        fallback_cfg["max_rel_diff"]
    )
    if got_fallback == "silent_eager_fallback":
        out["fallback_flagged"] = 1.0
    else:
        out["_note"] = f"expected 'silent_eager_fallback', got '{got_fallback}'"

    reassoc_cfg = ref.CONFIGS[0]
    got_reassoc = classify_run(
        reassoc_cfg["samples"],
        reassoc_cfg["baseline_med"],
        reassoc_cfg["baseline_mad"],
        reassoc_cfg["max_rel_diff"]
    )
    if got_reassoc == "reassociation" or got_reassoc == "normal":
        out["reassociation_handled"] = 1.0
    else:
        if "_note" not in out:
            out["_note"] = f"expected 'reassociation' or 'normal', got '{got_reassoc}'"

    return out
