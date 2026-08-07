import ref


def check(workdir):
    from kvtrace.simulator import simulate_peak_blocks

    out = {"traces_matched": 0.0, "total_traces": float(len(ref.TRACES))}
    ok = 0
    for i, (events, block_size) in enumerate(ref.TRACES):
        want = ref.simulate_peak_blocks(events, block_size)
        try:
            got = simulate_peak_blocks(events, block_size)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"trace {i}: got {got}, expected {want}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"trace {i} raised {type(e).__name__}: {str(e)[:100]}"

    out["traces_matched"] = float(ok)
    return out
