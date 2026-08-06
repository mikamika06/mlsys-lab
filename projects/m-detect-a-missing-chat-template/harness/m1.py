import ref


def check(workdir):
    from calib.detector import detect_chat_template

    out = {"templates_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.detect_chat_template(cfg) if hasattr(ref, "detect_chat_template") else None
        # fallback evaluation if ref doesn't have it locally
        if want is None:
            t = cfg.get("chat_template")
            want = bool(t and isinstance(t, str) and len(t.strip()) > 0 and ("{%" in t or "{{" in t))
        got = detect_chat_template(cfg)
        if bool(got) == bool(want):
            ok += 1
    out["templates_matched"] = float(ok)
    return out
