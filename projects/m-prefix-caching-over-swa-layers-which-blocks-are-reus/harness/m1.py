import ref


def check(workdir):
    from prefixcache.reuse import find_reusable_blocks

    out = {"reusable_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.find_reusable_blocks(cfg)
        got = find_reusable_blocks(cfg)
        if got == want:
            ok += 1
    out["reusable_matched"] = float(ok)
    return out
