import ref


def _gate_matches(got, want):
    if not isinstance(got, list) or len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if not isinstance(g, dict):
            return False
        if g.get("scheme") != w["scheme"] or g.get("bytes") != w["bytes"] or g.get("native") != w["native"]:
            return False
    return True


def check(workdir):
    from diskplan import best_native_scheme, gate_table

    out = {"gate_matches": 0.0, "best_matches": 0.0, "always_found": 0.0}
    total = float(len(ref.CASES))
    gate_ok = 0
    best_ok = 0
    found_ok = 0
    for i, (model, schemes, hardware) in enumerate(ref.CASES):
        want_gate = ref.gate_table(model, hardware, schemes)
        want_best = ref.best_native_scheme(model, hardware, schemes)
        try:
            got_gate = gate_table(model, hardware, schemes)
        except Exception as e:
            got_gate = None
            if "_note" not in out:
                out["_note"] = f"case {i} gate_table raised {type(e).__name__}: {str(e)[:120]}"
        try:
            got_best = best_native_scheme(model, hardware, schemes)
        except Exception as e:
            got_best = "<error>"
            if "_note" not in out:
                out["_note"] = f"case {i} best_native_scheme raised {type(e).__name__}: {str(e)[:120]}"
        if _gate_matches(got_gate, want_gate):
            gate_ok += 1
        if got_best == want_best:
            best_ok += 1
        if got_best is not None and got_best != "<error>":
            found_ok += 1
    out["gate_matches"] = gate_ok / total
    out["best_matches"] = best_ok / total
    out["always_found"] = found_ok / total
    return out
