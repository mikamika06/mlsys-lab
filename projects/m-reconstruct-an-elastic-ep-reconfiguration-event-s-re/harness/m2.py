import ref


def check(workdir):
    from eplb.reconstruct import reconstruct_layout
    out = {"trajectories_matched": 0.0, "configs": float(len(ref.BATCH_EVENTS))}
    ok = 0
    for init, events in ref.BATCH_EVENTS:
        want = ref.reconstruct_layout(init, events)
        got = reconstruct_layout(init, events)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trajectory failed: got {got}, want {want}"
    out["trajectories_matched"] = float(ok)
    return out
