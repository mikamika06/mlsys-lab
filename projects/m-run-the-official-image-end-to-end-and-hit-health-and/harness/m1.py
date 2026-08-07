import ref


def check(workdir):
    from vllm_runner.config import build_command

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_command(cfg)
        got = build_command(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
