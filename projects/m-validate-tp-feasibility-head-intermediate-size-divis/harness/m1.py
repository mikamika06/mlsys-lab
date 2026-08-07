import ref


def check(workdir):
    from tpval.feasibility import validate_tp_feasibility

    total = 0
    matched = 0

    for cfg in ref.CONFIGS:
        for tp in ref.TP_DEGREES:
            total += 1
            want = ref.validate_tp_feasibility(cfg, tp)
            got = validate_tp_feasibility(cfg, tp)
            if got.get("is_feasible") == want.get("is_feasible"):
                if got.get("is_feasible") or len(got.get("reasons", [])) > 0:
                    matched += 1

    feasibility_matches = 1.0 if total > 0 and matched == total else 0.0
    out = {
        "feasibility_matches": feasibility_matches,
        "total_evaluated": float(total),
        "matched": float(matched)
    }
    if feasibility_matches < 1.0:
        out["_note"] = f"Matched {matched}/{total} feasibility evaluations."
    return out
