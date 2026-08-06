import ref


def check(workdir):
    from pack.export import check_export_compatibility

    out = {"export_verified": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.verify_export(cfg)
        got = check_export_compatibility(cfg)
        if bool(got) == bool(want):
            ok += 1
    if ok == len(ref.CONFIGS):
        out["export_verified"] = 1.0
    return out
