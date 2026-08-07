import ref


def check(workdir):
    from ringbal import assign

    out = {"configs_matched": 0.0}
    matched = 0
    funcs = ["naive_assignment", "striped_assignment", "zigzag_assignment"]

    for N, D in ref.CONFIGS:
        for fn_name in funcs:
            got = getattr(assign, fn_name)(N, D)
            want = getattr(ref, fn_name)(N, D)
            if got == want:
                matched += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"{fn_name}({N}, {D}) mismatch: got {got[:2]}, want {want[:2]}"

    out["configs_matched"] = float(matched)
    return out
