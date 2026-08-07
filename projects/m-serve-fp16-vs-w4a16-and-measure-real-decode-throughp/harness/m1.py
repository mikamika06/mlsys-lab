import ref


def check(workdir):
    from serve.config import make_config

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.make_config(
            cfg["model_name"], cfg["quant_format"], cfg["batch_size"], cfg["seq_len"]
        )
        got = make_config(
            cfg["model_name"], cfg["quant_format"], cfg["batch_size"], cfg["seq_len"]
        )
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["configs_matched"] = float(ok)
    return out
