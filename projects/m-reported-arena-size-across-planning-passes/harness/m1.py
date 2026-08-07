import ref


def check(workdir):
    from arena.passes import track_arena_sizes

    out = {"passes_matched": 0.0, "configs": float(len(ref.PASSES_LIST))}
    ok = 0
    for i, passes in enumerate(ref.PASSES_LIST):
        want = ref.track_arena_sizes(passes)
        got = track_arena_sizes(passes)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"passes {i}: got {got}, reference {want}"
    out["passes_matched"] = float(ok)
    return out
