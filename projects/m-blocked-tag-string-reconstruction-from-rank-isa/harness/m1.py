import ref


def check(workdir):
    from dnnfmt.tags import reconstruct_tag

    out = {"tags_matched": 0.0, "total": float(len(ref.CASES))}
    ok = 0
    for i, case in enumerate(ref.CASES):
        got = reconstruct_tag(case["rank"], case["isa"])
        if got == case["want"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {case['want']}"
    out["tags_matched"] = float(ok)
    return out
