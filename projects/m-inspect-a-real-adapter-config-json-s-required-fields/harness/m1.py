import ref


def check(workdir):
    from peft_mechanics.config import inspect_config

    out = {"configs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.inspect_config(cfg)
        try:
            got = inspect_config(cfg)
            if got == want:
                ok += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"cfg {i}: want {want}, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"cfg {i} crashed: {e}"

    out["configs_matched"] = float(ok)
    return out
