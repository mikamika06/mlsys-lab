import ref


def check(workdir):
    from compilerdiag import diagnose

    out = {"layers_identified": 0.0, "scenarios": float(len(ref.SCENARIOS))}
    ok = 0
    for i, s in enumerate(ref.SCENARIOS):
        want = s["layer"]
        got = diagnose.identify_layer(s["id"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, reference {want}"
    out["layers_identified"] = float(ok)
    return out
