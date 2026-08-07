import ref


def check(workdir):
    from profops.ops import matmul_family_share
    out = {"share_matched": 0.0}
    ok = 0
    for table in ref.TABLES:
        want = ref.matmul_share(table)
        got = matmul_family_share(table)
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.TABLES):
        out["share_matched"] = 1.0
    return out
