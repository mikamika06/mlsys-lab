import ref


def check(workdir):
    from launchgraph.launches import count_launches

    out = {"launches_matched": 0.0}
    ok = 0
    total = len(ref.TEST_CONFIGS)

    for i, cfg in enumerate(ref.TEST_CONFIGS):
        want = ref.count_launches(**cfg)
        try:
            got = count_launches(**cfg)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"cfg {i} raised {type(e).__name__}: {e}"

    if ok == total:
        out["launches_matched"] = 1.0

    return out
