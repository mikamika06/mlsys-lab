import ref


def check(workdir):
    from ggml_isa.parser import parse_isa_flags

    out = {"flags_matched": 0.0, "logs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.parse_isa_flags(cfg["log"])
        got = parse_isa_flags(cfg["log"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log {i}: got {got}, reference {want}"
    out["flags_matched"] = float(ok)
    return out
