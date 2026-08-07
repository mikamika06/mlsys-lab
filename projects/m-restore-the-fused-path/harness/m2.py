import ref


def check(workdir):
    from flashfix.kernel import audit_contiguity

    out = {"contiguity_match": 0.0, "audit_precision": 0.0}
    want = ref.audit_contiguity(ref.CONFIGS)
    try:
        got = audit_contiguity(ref.CONFIGS)
    except Exception as e:
        out["_note"] = f"audit crashed: {str(e)[:100]}"
        return out

    matches = 0
    for g, w in zip(got, want):
        if g.get("layer_id") == w["layer_id"] and g.get("contiguous") == w["contiguous"]:
            matches += 1

    precision = float(matches) / float(len(want))
    out["audit_precision"] = precision
    if precision >= 0.95:
        out["contiguity_match"] = 1.0
    return out
