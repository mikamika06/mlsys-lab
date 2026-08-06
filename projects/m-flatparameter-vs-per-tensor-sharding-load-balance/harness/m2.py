import ref


def check(workdir):
    from fsdp_balance.sharding import auto_wrap_assign
    out = {"units_matched": 0.0, "total": float(len(ref.MODULE_TREES))}
    ok = 0
    for tree in ref.MODULE_TREES:
        min_p = 300
        want = ref.auto_wrap_assign(tree, min_p)
        try:
            got = auto_wrap_assign(tree, min_p)
            if sorted(got) == want:
                ok += 1
        except Exception:
            pass
    out["units_matched"] = float(ok)
    return out
