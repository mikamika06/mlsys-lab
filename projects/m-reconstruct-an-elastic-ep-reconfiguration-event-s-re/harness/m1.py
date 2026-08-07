import ref


def check(workdir):
    from eplb.reconstruct import apply_event
    out = {"events_matched": 0.0, "configs": float(len(ref.SINGLE_EVENTS))}
    ok = 0
    for event, layout in ref.SINGLE_EVENTS:
        want = ref.apply_event(layout, event)
        got = apply_event(layout, event)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"event {event}: got {got}, want {want}"
    out["events_matched"] = float(ok)
    return out
