import ref


def check(workdir):
    from profops.ops import top_1_operator_by_self_time
    out = {"top1_matched": 0.0}
    ok = 0
    for table in ref.TABLES:
        want = ref.top_1_operator(table)
        got = top_1_operator_by_self_time(table)
        if got == want:
            ok += 1
    if ok == len(ref.TABLES):
        out["top1_matched"] = 1.0
    return out
