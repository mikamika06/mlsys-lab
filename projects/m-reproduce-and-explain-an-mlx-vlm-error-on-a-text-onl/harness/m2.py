import ref


def check(workdir):
    from mlx_vlm_edge.token_counter import compute_image_tokens

    out = {"token_counts_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        try:
            got = compute_image_tokens(cfg["resolution"], cfg["patch_size"], cfg.get("vision_config"))
            if got == cfg["want"]:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, want {cfg['want']}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised {type(e).__name__}: {str(e)}"
    out["token_counts_matched"] = float(ok)
    return out
