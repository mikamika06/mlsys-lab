import ref


def check(workdir):
    from nmval.validator import validate_nm_constraint

    out = {"validators_matched": 0.0, "total": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_valid, want_counts = ref.validate_nm(cfg, 2, 4)
        got_valid, got_counts = validate_nm_constraint(cfg, 2, 4)
        if got_valid == want_valid and got_counts == want_counts:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got ({got_valid}, {got_counts}), reference ({want_valid}, {want_counts})"
    out["validators_matched"] = float(ok)
    return out
