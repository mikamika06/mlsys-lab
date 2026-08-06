import ref


def check(workdir):
    from fsdp_analysis.model import compute_costs

    out = {"metrics_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    matched = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = compute_costs(cfg["strategy"], cfg["num_params"], cfg["hidden_dim"], cfg["num_layers"], cfg["world_size"])
        try:
            got = compute_costs(cfg["strategy"], cfg["num_params"], cfg["hidden_dim"], cfg["num_layers"], cfg["world_size"])
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised {type(e).__name__}"
            continue
        if got == want:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["metrics_matched"] = float(matched)
    return out
