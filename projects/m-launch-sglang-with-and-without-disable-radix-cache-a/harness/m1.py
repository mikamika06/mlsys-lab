import ref


def check(workdir):
    from sgl_utils import launcher
    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_launch_command(
            model_path=cfg["model_path"],
            port=cfg["port"],
            disable_radix_cache=cfg["disable_radix_cache"],
            extra_args=cfg["extra_args"]
        )
        got = launcher.build_launch_command(
            model_path=cfg["model_path"],
            port=cfg["port"],
            disable_radix_cache=cfg["disable_radix_cache"],
            extra_args=cfg["extra_args"]
        )
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
