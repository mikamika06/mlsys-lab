import ref


def check(workdir):
    from realign.aligner import align_tokens

    out = {"alignments_matched": 0.0}
    ok = 0
    for case in ref.CASES:
        got = align_tokens(case["draft"], case["vocab"])
        if got == case["expected"]:
            ok += 1
    out["alignments_matched"] = float(ok)
    return out
