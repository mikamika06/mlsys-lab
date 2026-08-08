import ref


def check(workdir):
    from jaxpr_utils.analyzer import count_equations

    matched = 0
    for i, jaxpr in enumerate(ref.SAMPLE_JAXPRS):
        want = ref.count_equations(jaxpr)
        got = count_equations(jaxpr)
        if want == got:
            matched += 1

    out = {"eqn_counts_matched": 1.0 if matched == len(ref.SAMPLE_JAXPRS) else 0.0}
    if matched != len(ref.SAMPLE_JAXPRS):
        out["_note"] = f"Matched {matched}/{len(ref.SAMPLE_JAXPRS)} jaxprs"
    return out
