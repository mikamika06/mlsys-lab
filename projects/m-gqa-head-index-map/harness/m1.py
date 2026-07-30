import ref


def check(workdir):
    from gqa.mapping import build_head_map, build_query_groups

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, (nq, nkv) in enumerate(ref.CONFIGS):
        want_map = ref.build_head_map(nq, nkv)
        want_groups = [sorted(g) for g in ref.build_query_groups(nq, nkv)]
        try:
            got_map = list(build_head_map(nq, nkv))
            got_groups = [sorted(g) for g in build_query_groups(nq, nkv)]
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"config {i} ({nq},{nkv}): raised {type(e).__name__}: {str(e)[:120]}"
            continue
        if got_map == want_map and got_groups == want_groups:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (f"config {i} ({nq},{nkv}): got map={got_map} groups={got_groups}, "
                             f"want map={want_map} groups={want_groups}")
    out["configs_matched"] = float(ok)
    return out
