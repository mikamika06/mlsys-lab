import ref


def check(workdir):
    from autotune.searchspace import compute_search_space_size

    out = {"search_space_matched": 0.0}
    ok = 0
    for i, (tiles, warps, stages) in enumerate(ref.SPACES):
        count = 0
        for t in tiles:
            for w in warps:
                for s in stages:
                    if t[0] >= 16 and t[1] >= 16 and w >= 2 and s >= 1:
                        count += 1
        try:
            got = compute_search_space_size(tiles, warps, stages)
            if int(got) == int(count):
                ok += 1
        except Exception:
            pass
    out["search_space_matched"] = float(ok)
    return out
