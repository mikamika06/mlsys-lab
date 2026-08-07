import ref


def check(workdir):
    from striped.policy import assign_blocks

    out = {"policies_matched": 0.0}
    ok = 0
    for i, scn in enumerate(ref.SCENARIOS):
        nb = scn["num_blocks"]
        ws = scn["world_size"]
        want = ref.assign_blocks(nb, ws)
        got = assign_blocks(nb, ws)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, reference {want}"
    out["policies_matched"] = float(ok)
    return out
