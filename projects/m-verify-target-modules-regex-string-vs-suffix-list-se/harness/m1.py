import ref


def check(workdir):
    from loratarget.matcher import resolve_by_regex, resolve_by_suffix

    out = {"matched_configs": 0.0, "total_configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, (tree, pattern, suffixes) in enumerate(ref.CONFIGS):
        want_regex = ref.resolve_by_regex(tree, pattern)
        want_suffix = ref.resolve_by_suffix(tree, suffixes)

        got_regex = resolve_by_regex(tree, pattern)
        got_suffix = resolve_by_suffix(tree, suffixes)

        if sorted(got_regex) == sorted(want_regex) and sorted(got_suffix) == sorted(want_suffix):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {i}: regex got {got_regex[:2]}, want {want_regex[:2]}; suffix got {got_suffix[:2]}, want {want_suffix[:2]}"

    out["matched_configs"] = float(ok)
    return out
