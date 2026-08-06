import ref


def check(workdir):
    from ortopt.blocklist import get_minimal_blocklist

    out = {"blocklist_match": 0.0}
    ok = 0
    for model_type in ref.CONFIGS:
        want = ref.get_minimal_blocklist(model_type)
        got = get_minimal_blocklist(model_type)
        if sorted(got or []) == sorted(want):
            ok += 1
    if ok == len(ref.CONFIGS):
        out["blocklist_match"] = 1.0
    return out
